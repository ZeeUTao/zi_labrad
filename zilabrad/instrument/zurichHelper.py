import textwrap  # to write sequencer's code
import time  # show total time in experiments
import enum
import zhinst.utils  # create API object
import numpy as np
from numpy import pi
from functools import wraps
import logging
import os

from zilabrad.util import singleton, singletonMany
from zilabrad.instrument.waveforms import convertUnits


def create_logger(name, filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)

    filepath = os.path.join(os.getcwd(), filename+'.log')
    formatter = logging.Formatter(
        '%(asctime)s  %(name)s \n%(levelname)s: %(message)s\n')
    fileHandler = logging.FileHandler(filepath, encoding='UTF-8', mode='w')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    return logger


logger = create_logger(__name__, __name__)


def get_QA_program(
        sample_rate, number_port, wave_length, *args, **kwargs):
    """awg program for labone
    Example:
    awg_program = get_QA_program(sample_rate=int(1.8e9), number_port=2)
    """
    wave_define_string = ""
    wave_play_string = ""
    for i in range(number_port):
        wave_define_string += "wave w" + str(i+1) +\
            "= zeros(" + str(wave_length) + ");\n"
        wave_play_string += (',' + str(i+1) + ',w'+str(i+1))
    wave_play_string = wave_play_string[1:]
    # delect the first comma

    awg_program = textwrap.dedent("""\
const f_s = $sample_rate;
$wave_define_string
setTrigger(AWG_INTEGRATION_ARM);// initialize integration
setTrigger(0b000);
repeat (getUserReg(0)) {  // = qa.result_samples
        // waitDigTrigger(1,1);
        setTrigger(0b11); // trigger output: rise
        wait(5); // trigger length: 22.2 ns / 40 samples
        setTrigger(0b00); // trigger output: fall
        playWave($wave_play_string);
        wait(getUserReg(1)); // demod wait time -> qa demod start
        setTrigger(AWG_INTEGRATION_ARM + AWG_INTEGRATION_TRIGGER + \
AWG_MONITOR_TRIGGER);// start demodulate
        setTrigger(AWG_INTEGRATION_ARM);// reset intergration
        waitWave();
        wait(getUserReg(2)); // wait relaxation (default 200us)
}
    """)
    awg_program = awg_program.replace(
        '$sample_rate', str(sample_rate))
    awg_program = awg_program.replace(
        '$wave_define_string', wave_define_string)
    awg_program = awg_program.replace(
        '$wave_play_string', wave_play_string)
    return awg_program


def get_HD_program(
        sample_rate: int, number_port: int, wave_length: int, loop=False):
    """
    Return (str):
        awg program for labone
    """
    def raw_program(
        FS, define_str, play_str, loop
    ):
        if loop:
            trigger_str = '//waitDigTrigger(1);'
        else:
            trigger_str = 'waitDigTrigger(1);'
        program = textwrap.dedent(f"""\
const f_s = {FS};
{define_str}
while(1){"{"}
{trigger_str}
playWave({play_str});
waitWave();
{"}"}
""")
        return program

    def wave_define_func(idx):
        return f'wave w{idx} = zeros({wave_length});\n'

    ports_array = np.arange(1, number_port+1, 1)
    # O(n) for join,  str += str1+str2 can be O(n^2)
    wave_define_str = "".join(
        list(map(wave_define_func, ports_array))
    )
    wave_play_list = list(
        map(lambda idx: f'{idx},w{idx},', ports_array)
    )
    # remove last comma ","
    wave_play_list[-1] = wave_play_list[-1][:-1]
    wave_play_str = "".join(wave_play_list)

    awg_program = raw_program(
        sample_rate, wave_define_str, wave_play_str, loop)
    return awg_program


def convert_awg_waveform(wave_list):
    """
    Converts one or multiple arrays with waveform data to the native AWG
    waveform format (interleaved waves and markers as uint16).
    Waveform data can be provided as integer (no conversion) or floating point
    (range -1 to 1) arrays.
    Arguments:
      wave1 (array): Array with data of waveform 1.
      wave2 (array): Array with data of waveform 2.
      ...
    Returns:
      The converted uint16 waveform is returned.
    NOTE: waveform reload needs each two channel one by one.
    """
    def uint16_waveform(wave):  # Prepare waveforms
        wave = np.asarray(wave)
        if np.issubdtype(wave.dtype, np.floating):
            return np.asarray((np.power(2, 15) - 1) * wave, dtype=np.uint16)
        return np.asarray(wave, dtype=np.uint16)

    data_tuple = (uint16_waveform(wave_list[0]), uint16_waveform(wave_list[1]))
    return np.vstack(data_tuple).reshape((-2,), order='F')


class qaSource(enum.Enum):
    """ Constants (int) for selecting result logging source """
    TRANS = 0
    THRES = 1
    ROT = 2
    TRANS_STAT = 3
    CORR_TRANS = 4
    CORR_THRES = 5
    CORR_STAT = 6
    INTEGRATION = 7
    name = [
        'TRANS', 'THRES', 'ROT', 'TRANS_STAT', 'CORR_TRANS',
        'CORR_THRES', 'CORR_STAT', 'INTEGRATION'
    ]


@singleton
class ziDAQ(object):
    """singleton class for zurich daq
    """
    def __init__(self,labone_ip='localhost'):
        # connectivity must 8004 for zurish instruments
        self.daq = zhinst.ziPython.ziDAQServer(labone_ip,8004,6)
        self.secret_mode(mode=1) ## (default) daq can be used by everyone.

    def secret_mode(self, mode=1):
        # mode = 1: daq can be created by everyone
        # mode = 0: daq only be used by localhost
        self.daq.setInt('/zi/config/open', mode)

    def refresh_api(self,labone_ip):
        self.daq = zhinst.ziPython.ziDAQServer(labone_ip,8004,6)


@singletonMany
class zurich_qa(object):
    """server for zurich qa
    Args:
        obj_name (str): specify the object, the object with the same obj_name
        is singleton.
        Example: If the class has been instantiated with obj_name=='obj1',
        calling it again via 'zurich_qa(obj_name='obj1')' will not
        re-initiate but get the old object.
    Instance: The instance now will return a dictionary of objects (value)
    and their obj_name (key) that has been created
    """

    def __init__(self, obj_name='qa_1', device_id='dev2592',
                 labone_ip='localhost'):
        self.obj_name = obj_name
        self.id = device_id
        try:
            logger.info("\nBring up %s in %s" % (self.id, labone_ip))
            self.daq = ziDAQ(labone_ip=labone_ip).daq
            self.connect()
            logger.info(self.daq)
            self.FS = 1.8e9 / \
                (self.daq.getInt(
                    '/{:s}/awgs/0/time'.format(self.id))+1)  # sample rate
            self.init_setup()
        except Exception as e:
            logger.error("Failed to initialize [%s]" % self.id.upper())
            raise e

    # -- set device status
    def connect(self):
        self.daq.connectDevice(self.id, '1gbe')

    def disconnect(self):
        self.daq.disconnectDevice(self.id)

    def refresh_api(self,labone_ip='localhost'):
        self.daq = ziDAQ(labone_ip=labone_ip).daq

    def init_setup(self):
        """ initialize device settings.
        """
        self.average = 1  # default 1, no average in device
        self.result_samples = 1024
        self.qubit_frequency = []  # all demodulate frequency; unit: Hz
        self.paths = []  # save result path, equal to channel number
        # qa pulse length in AWGs; unit: sample number
        self.waveform_length = 0
        # qa integration length; unit: sample number
        self.integration_length = 4096
        # qa result mode: integration--> return origin (I+iQ)
        self.source = qaSource.INTEGRATION.value

        ## set experimental parameters ##
        self.set_relaxation_length(relax_time=200e-6) ## unit: second
        # self.set_demod_start(0) # set demod start delay after All device trigger
        # self.set_demod_length(0) # set self.integration_length

        # set qa demod module
        self.set_result_samples() ## (default input) self.result_samples
        self.set_qaSource_mode() ## (default input) self.source
        # set integration mode: 0=standard mode, 1=spectroscopy mode;
        self.daq.setInt('/{:s}/qas/0/integration/mode'.format(self.id), 0)

        # set signal output (readin line)
        # awgs output mode: 0=plain, 1=modulation; 
        self.daq.setInt('/{:s}/awgs/0/outputs/*/mode'.format(self.id), 0)
        # awgs hold maximum amplitude 1
        self.daq.setDouble('/{:s}/awgs/0/outputs/*/amplitude', 1)
        self.daq.setInt('/{:s}/sigouts/*/imp50', 0) ## close 50 Ohm output impedance
        self.port_range(1.5,port=[1,2]) ## awg output range 1.5V(default);

        # set signal input (readout line)
        # input range 1V;  Range=[10mV,1.5V]
        self.daq.setDouble('/{:s}/sigins/*/range'.format(self.id), 1)
        # 50 Ohm input impedance (must open)
        self.daq.setInt('/{:s}/sigins/*/imp50'.format(self.id), 1)

        # set trigger part
        # open trig 1
        self.daq.setInt('/{:s}/triggers/out/0/source'.format(self.id), 32)
        # open trig 2
        self.daq.setInt('/{:s}/triggers/out/1/source'.format(self.id), 33)
        self.daq.setInt('/{:s}/triggers/out/*/drive'.format(self.id), 1)
        # set DIO output as qubit result
        self.daq.setInt('/{:s}/dios/0/drive'.format(self.id), 15)
        self.daq.setInt('/{:s}/dios/0/mode'.format(self.id), 2)
        # Sample DIO data at 50 MHz
        self.daq.setInt('/{:s}/dios/0/extclk'.format(self.id), 2)

        ## open AWGs port output 
        self.port_output(output=True)
        # close Rerun
        self.daq.setInt('/{:s}/awgs/0/single'.format(self.id), 1)
        logger.info('%s: Complete Initialization' % self.id)

    # -- set AWGs status
    def update_pulse_length(self):
        ''' Update self.waveform_length value via 
            daq.getList(), make sure send_waveform()
            having true length.
        '''
        qainfo = self.daq.getList(
            '/{:s}/awgs/0/waveform/waves/0'.format(self.id))
        if len(qainfo) == 0:
            self.waveform_length = 0
        elif len(qainfo) == 1:
            # qalen has double channel wave;
            self.waveform_length = int(len(qainfo[0][1][0]['vector'])/2)
        else:
            raise Exception('Unknown QA infomation:\n', qainfo)
        logger.info(
            '[%s] update_pulse_length: %r' % (self.id, self.waveform_length))

    def awg_open(self):
        self.daq.syncSetInt('/{:s}/awgs/0/enable'.format(self.id), 1)
        logger.debug('\n AWG running. \n')

    def awg_close(self):
        # Stop result unit
        self.daq.unsubscribe(self.paths)
        self.daq.setInt('/{:s}/qas/0/result/enable'.format(self.id), 0)

    def port_output(self,output=True,port=[1,2]):
        # output = 1, open AWG port; output = 0, close AWG port;
        if isinstance(port, int):
            port = [port]
        for p in port:
            self.daq.setInt('/{:s}/sigouts/{:d}/on'.format(self.id,int(p-1)), int(output))

    @convertUnits(range_='V')
    def port_range(self,range_=None,port=[1,2]):
        # AWG output range = 1.5V or 150mV
        # range_=None -> return current range from ZI device
        if range_ == None:
            self.range = [None,None]
            for p in range(2):
                self.range[p] = round(self.daq.getDouble(
                                    '/{:s}/sigouts/{:d}/range'.format(self.id,int(p))),5)
            return self.range
        else:
            if isinstance(port, int):
                port = [port]
            for p in port:
                self.daq.setDouble('/{:s}/sigouts/{:d}/range'.format(self.id,int(p-1)), range_)

    # -- send AWGs waveform
    def _awg_builder(self, number_port, wave_length, awg_index=0):
        """ Build waveforms sequencer. Then compile and send it to devices.
        """
        logger.info(
            'Bulid [%s-AWG0] Sequencer (len=%r > %r)' %
            (self.id, wave_length, self.waveform_length))
        t0 = time.time()
        # create default zeros waveform
        awg_program = get_QA_program(
            sample_rate=int(self.FS),
            number_port=number_port,
            wave_length=wave_length)

        self._awg_upload_string(awg_program, awg_index=awg_index)
        self.update_pulse_length()  # updata self.waveform_lenght
        logger.info(
            '[%s-AWG0] builder: %.3f s' % (self.id, time.time()-t0))

    def _awg_upload_string(self, awg_program, awg_index=0):
        """ awg_program: waveforms sequencer text
            awg_index: this device's awgs sequencer index. 
                       If awgs grouping == 4*2, this index 
                       can be selected as 0,1,2,3
            write into waveforms sequencer and compile it.
        """
        awgModule = self.daq.awgModule()  # this API needs 0.2s to create
        awgModule.set('awgModule/device', self.id)
        awgModule.set('awgModule/index', awg_index)
        awgModule.execute()  # Starts the awgModule if not yet running.
        awgModule.set('awgModule/compiler/sourcestring',
                      awg_program)  # to compile
        while awgModule.getInt('awgModule/compiler/status') == -1:
            time.sleep(0.1)
        # Ensure that compilation was successful
        if awgModule.getInt('awgModule/compiler/status') == 1:
            # compilation failed, raise an exception
            raise Exception(awgModule.getString(
                'awgModule/compiler/statusstring'))
        else:
            if awgModule.getInt('awgModule/compiler/status') == 0:
                logger.debug(
                    "Compilation successful with no warnings, \
                will upload the program to the instrument.")
            if awgModule.getInt('awgModule/compiler/status') == 2:
                logger.warning(
                    "Compilation successful with no warnings, \
                will upload the program to the instrument.")
                logger.warning("Compiler warning: ", awgModule.getString(
                    'awgModule/compiler/statusstring'))
            # wait for waveform upload to finish
            while awgModule.getDouble('awgModule/progress') < 1.0:
                time.sleep(0.1)
        logger.debug('\n AWG upload successful. Output enabled. AWG Standby.')

    def send_waveform(self, waveform, recursion=3):
        """
        Args:
            waveform: all waveform in this device
            e.g.: [[1.,1.,...],[1.,1.,...]]
            Here judge which awgs or port will be used
            to reload. Fill zeros at the end of waveform
            to match the prior waveform length or compile
            sequencer again.

            recursion (int): the function will be called
            at most (recursion+1) times
        """
        if recursion < 0:
            raise Exception("recursion callings exceed")

        _n_ = self.waveform_length - len(waveform[0])
        if _n_ < 0:
            self._awg_builder(
                number_port=len(waveform),
                wave_length=len(waveform[0]),
                awg_index=0)

            self.send_waveform(
                waveform=waveform,
                recursion=recursion-1)
            return
        else:
            waveform_add = [np.hstack((wf, np.zeros(_n_))) for wf in waveform]
            try:
                self._reload_waveform(waveform=waveform_add)
            except Exception:
                self.update_pulse_length()
                self.send_waveform(
                    waveform=waveform,
                    recursion=recursion-1)
            return

    def _reload_waveform(self, waveform, awg_index=0, index=0):
        """ waveform: (numpy.array) one/two waves with unit amplitude.
            awg_index: this devices awg sequencer index
            index: this waveform index in total sequencer
        """
        waveform_native = convert_awg_waveform(waveform)
        path = '/{:s}/awgs/{:d}/waveform/waves/{:d}'.format(
            self.id, awg_index, index)
        self.daq.setVector(path, waveform_native)

    # -- set qa demod parameters
    @convertUnits(readout_delay='s')
    def set_relaxation_length(self,relax_time):
        # send to device: Register 3
        self.daq.setDouble(
                    '/{:s}/awgs/0/userregs/2'.format(self.id), 
                    int(relax_time*self.FS/8))

    def set_result_samples(self, sample=None):
        """ sample: repetition number
            Meanwhile update repeat index in AWG sequencer
            and QA result parameter.
        """
        if sample != None:
            self.result_samples = sample
        # send to device: Register 1
        self.daq.setDouble(
                    '/{:s}/awgs/0/userregs/0'.format(self.id),
                    self.result_samples)
        self.daq.setInt('/{:s}/qas/0/result/length'.format(self.id),
                        self.result_samples)  # results length

    @convertUnits(readout_delay='s')
    def set_demod_start(self,demod_start):
        ''' demod_start: All device trigger --> QA integration start
            Here convert value from second to sample number, 
            8 samples as a unit.
        '''
        # unit: Sample Number
        # send to device: Register 1
        self.daq.setInt(
            '/{:s}/awgs/0/userregs/1'.format(self.id), int(demod_start*self.FS/8)*8)

    @convertUnits(length='s')
    def set_demod_length(self, length):
        ''' length: set length for demodulate.
            INPUT: unit --> Second,
            SAVE: unit --> Sample number
            Demodulate has maximum length 4096.
            Ignore exceeding part.
        '''
        if int(length*self.FS) > 4096:
            logger.warning(
                'QA pulse lenght too long ( %.1f>%.1f ns)' %
                (length, 4096/1.8))
            self.waveform_length = 4096  # set the maximum length
        else:
        # unit --> Sample Number
        self.integration_length = int(length*self.FS/4)*4

    # -- set qa demod mode
    def set_qaSource_mode(self, mode=None):
        if mode == None:
            mode = self.source
        if mode == qaSource.INTEGRATION.value:
            self.daq.setInt(
                '/{:s}/qas/0/integration/sources/*'.format(self.id), 0)
            self.daq.setComplex('/{:s}/qas/0/rotations/*'.format(self.id), 1)
            self._set_deskew_matrix([[1, 1], [1, 1]])
        else:
            for k in range(5):
                self.daq.setInt(
                    '/{:s}/qas/0/integration/sources/{:d}'.format(
                        self.id, 2*k), 0)
                self.daq.setInt(
                    '/{:s}/qas/0/integration/sources/{:d}'.format(
                        self.id, 2*k+1), 1)
                self.daq.setComplex(
                    '/{:s}/qas/0/rotations/{:d}'.format(
                        self.id, 2*k), 1-1j)
                self.daq.setComplex(
                    '/{:s}/qas/0/rotations/{:d}'.format(
                        self.id, 2*k+1), 1+1j)
            self._set_deskew_matrix([[1, 0], [0, 1]])
        self.daq.setInt(
            '/{:s}/qas/0/result/source'.format(self.id), self.source)
        ## update 'self.source' info from ZI device
        self.source = self.daq.getInt(
            '/{:s}/qas/0/result/source'.format(self.id), self.source)        
        logger.info(
            ' [%s] QA Source Mode: %s' % (self.id, qaSource.name.value[self.source]))

    def _set_deskew_matrix(self, matrix=[[1, 1], [1, 1]]):
        self.daq.setDouble(
            '/{:s}/qas/0/deskew/rows/0/cols/0'.format(self.id), matrix[0][0])
        self.daq.setDouble(
            '/{:s}/qas/0/deskew/rows/0/cols/1'.format(self.id), matrix[0][1])
        self.daq.setDouble(
            '/{:s}/qas/0/deskew/rows/1/cols/0'.format(self.id), matrix[1][0])
        self.daq.setDouble(
            '/{:s}/qas/0/deskew/rows/1/cols/1'.format(self.id), matrix[1][1])

    # -- set qa demodulation frequency (qubit sideband part)
    def set_qubit_frequency(self, frequency_list):
        # set integration weights, and result paths
        self.qubit_frequency = np.zeros(10)
        n_ch = len(frequency_list)
        if self.source == qaSource.INTEGRATION.value:
            if n_ch > 10:
                logger.warning('frequency list(len=%d) exceeds the max \
                channel number 10.' % n_ch)
            self.qubit_frequency[0:n_ch:1] = frequency_list[:10]
        if self.source == qaSource.ROT.value:
            if n_ch > 5:
                logger.warning('frequency list(len=%d) exceeds the max \
                    channel number 5.' % n_ch)
            self.qubit_frequency[0:n_ch*2:2] = frequency_list[:5]
            self.qubit_frequency[1:n_ch*2:2] = frequency_list[:5]
        self._set_all_integration()  # set integration weights
        self._set_subscribe()  # set result paths

    def _set_all_integration(self):
        w_index = np.arange(0, self.integration_length, 1)
        for channel, freq in enumerate(self.qubit_frequency):
            # assign real and image integration coefficient
            # integration settings for one I/Q pair
            self.daq.setDouble(
                '/{:s}/qas/0/integration/length'.format(self.id),
                self.integration_length)

            w_real = np.cos(w_index*freq/1e9/1.8*2*pi)
            w_imag = np.sin(w_index*freq/1e9/1.8*2*pi)

            self.daq.setVector(
                '/{:s}/qas/0/integration/weights/{}/real'.format(
                    self.id, channel), w_real)
            self.daq.setVector(
                '/{:s}/qas/0/integration/weights/{}/imag'.format(
                    self.id, channel), w_imag)

    def _set_subscribe(self, source=None):
        """ set demodulate result parameters -> upload qa result's paths
        """
        if source is None:
            source = self.source
        self.daq.setInt('/{:s}/qas/0/result/length'.format(self.id),
                        self.result_samples)  # results length
        # average results
        self.daq.setInt(
            '/{:s}/qas/0/result/averages'.format(self.id), self.average)

        # reset qa result
        self.daq.setInt('/{:s}/qas/0/result/reset'.format(self.id), 1)
        # stert qa result module, wait value
        self.daq.setInt('/{:s}/qas/0/result/enable'.format(self.id), 1)

        def get_path(ch): 
            return '/{:s}/qas/0/result/data/{:d}/wave'.format(self.id, ch)
        chs = range(len(self.qubit_frequency))
        self.paths = list(map(get_path, chs))

        logger.debug(f'Subscribed paths: {self.paths}')
        self.daq.subscribe(self.paths)

    # -- get demod result
    def _acquisition_poll(self, daq, paths, num_samples, timeout=10.0):
        """ Polls the UHFQA for data.
        Args:
            paths (list): list of subscribed paths
            num_samples (int): expected number of samples
            timeout (float): time in seconds before timeout Error is raised.
        """
        logger.debug('acquisition_poll')
        poll_length = 0.01  # s
        poll_timeout = 100  # ms
        poll_flags = 0
        poll_return_flat_dict = True

        # Keep list of recorded chunks of data for each subscribed path
        chunks = {p: [] for p in paths}
        gotem = {p: False for p in paths}

        # Poll data
        time = 0

        def get_vec(v): return v['vector']

        while time < timeout and not all(gotem.values()):
            logger.debug('collecting results')
            dataset = daq.poll(poll_length, poll_timeout,
                               poll_flags, poll_return_flat_dict)
            for p in paths:
                if p not in dataset:
                    continue
                chunks[p] = list(map(get_vec, dataset[p]))
                num_obtained = sum(map(len, chunks[p]))
                if num_obtained >= num_samples:
                    gotem[p] = True
            time += poll_length

        if not all(gotem.values()):
            for p in paths:
                num_obtained = sum(map(len, chunks[p]))
                logger.error('Path {}: Got {} of {} samples'.format(
                    p, num_obtained, num_samples))
            raise Exception(
                'Timeout Error: Did not get all results \
                within {:.1f} s!'.format(timeout))

        # Return dict of flattened data
        return {p: np.concatenate(v) for p, v in chunks.items()}

    def get_data(self):
        data = self._acquisition_poll(
            self.daq, self.paths, self.result_samples, timeout=10)
        return list(data.values())


@singletonMany
class zurich_hd:
    """server for zurich hd
    Args:
        obj_name (str): specify the object, the object with the same obj_name
        is singleton.
        Example: If the class has been instantiated with obj_name=='obj1',
        calling it again via 'zurich_hd(obj_name='obj1')' will not
        re-initiate but get the old object.
    Instance: The instance now will return a dictionary of objects (value)
    and their obj_name (key) that has been created
    """

    def __init__(self, obj_name='hd_1', device_id='dev8334',
                 labone_ip='localhost'):
        self.id = device_id
        self.obj_name = obj_name
        try:
            logger.info('\nBring up %s in %s' % (self.id, labone_ip))
            self.daq = ziDAQ(labone_ip=labone_ip).daq
            self.connect()
            logger.info(self.daq)
            # sample rate
            self.FS = self.daq.getDouble(
                '/{:s}/system/clocks/sampleclock/freq'.format(self.id))
            self.init_setup()
        except Exception as e:
            logger.error("Failed to initialize [%s]" % self.id.upper())
            raise e

    # -- set device status
    def connect(self):
        self.daq.connectDevice(self.id, '1gbe')

    def disconnect(self):
        self.daq.disconnectDevice(self.id)

    def refresh_api(self,labone_ip='localhost'):
        self.daq = ziDAQ(labone_ip=labone_ip).daq

    def init_setup(self):
        # four awg's waveform length, unit --> Sample Number
        self.waveform_length = [0, 0, 0, 0]
        self.update_pulse_length() ## update current 'waveform_length' from ZI device
        self.port_output(output=True) # open all signal output port
        self.port_range(range_=1) # default output range: 1V

        self.awg_grouping(grouping_index=0) # (default) index == 0 --> 4*2 grouping mode
        # [AWG 0~3] digital trigger 1&2: slope --> rise
        self.daq.setInt('/{:s}/awgs/*/auxtriggers/*/slope'.format(self.id), 1)
        # (DIO) trigger in, set 50 Ohm impedance
        self.daq.setInt('/{:s}/triggers/in/*/imp50'.format(self.id), 0)

        # use maximum sample rate in AWG sequence
        self.daq.setInt('/{:s}/awgs/*/time'.format(self.id), 0)
        # set ref clock mode as 'External'
        self.daq.setInt('/{:s}/system/clocks/referenceclock/source'.format(self.id), 1) 
        # awgs hold maximum amplitude 1
        self.daq.setInt('/{:s}/awgs/0/outputs/*/amplitude'.format(self.id), 1.0) 
        # awgs output mode: 0=plain, 1=modulation; 
        self.daq.setInt('/{:s}/awgs/0/outputs/0/modulation/mode'.format(self.id), 0)  

        self._unknown_settings()

        # close rerun
        self.daq.setInt('/{:s}/awgs/0/single'.format(self.id), 1)
        # Ensure that all settings have taken effect on the device before continuing.
        ### self.daq.sync()
        logger.info('%s: Complete Initialization' % self.id.upper())

    def _unknown_settings(self)
        ## Unknown settings were suggested by ZI engineer
        self.daq.setInt('/%s/awgs/0/dio/strobe/slope'.format(self.id), 0)
        self.daq.setInt('/%s/awgs/0/dio/strobe/index'.format(self.id), 15)
        self.daq.setInt('/%s/awgs/0/dio/valid/index'.format(self.id), 0)
        self.daq.setInt('/%s/awgs/0/dio/valid/polarity'.format(self.id), 2)
        self.daq.setInt('/%s/awgs/0/dio/mask/value'.format(self.id), 7)  # 111 三位qubit results
        self.daq.setInt('/%s/awgs/0/dio/mask/shift'.format(self.id), 1)
        self.daq.setInt('/%s/raw/dios/0/extclk'.format(self.id), 2)

    # -- set awg status
    def awg_open(self, awgs_index=[0, 1, 2, 3]):
        # run specific AWG following awgs_index
        if isinstance(awgs_index, int):
            awgs_index = [awgs_index]
        for i in awgs_index:
            self.daq.setInt('/{:s}/awgs/{:d}/enable'.format(self.id, i), 1)
            logger.debug('%s AWG%d running.' % (self.id, i))

    def awg_close(self, awgs_index=[0, 1, 2, 3]):
        # stop specific awg following awgs_index
        if isinstance(awgs_index, int):
            awgs_index = [awgs_index]
        for i in awgs_index:
            self.daq.setInt('/{:s}/awgs/{:d}/enable'.format(self.id, i), 0)

    def awg_grouping(self, grouping_index=0):
        """ grouping_index:
                0 : 4x2 with HDAWG8; 2x2 with HDAWG4.
                1 : 2x4 with HDAWG8; 1x4 with HDAWG4.
                2 : 1x8 with HDAWG8.
            set AWG grouping mode, following path:
            '/dev_id/system/awg/channelgrouping', Configure
            how many independent sequencers, should run on
            the AWG and how the outputs are grouped by sequencer.
        """
        grouping_name = ['4x2 with HDAWG8', '2x4 with HDAWG8', '1x8 with HDAWG8']

        ## update 'grouping' info
        self.grouping = self.daq.getInt(
                '/{:s}/system/awg/channelgrouping'.format(self.id))

        ## try to set grouping mode
        if int(grouping_index) != int(self.grouping):
            self.daq.setInt(
                '/{:s}/system/awg/channelgrouping'.format(self.id), grouping_index)
            if grouping_index == 0:
                for awg in range(4):
                    # set digital trigger
                    self.daq.setInt(
                        '/{:s}/awgs/{:d}/auxtriggers/0/channel'.format(self.id, awg), int(awg*2))
                    # set trigger threshold
                    self.daq.setDouble(
                        '/{:s}/triggers/in/{:d}/level'.format(self.id, int(awg*2)), 0.1)
            if grouping_index == 1:
                for awg in range(2):
                    # set digital trigger
                    self.daq.setInt(
                        '/{:s}/awgs/{:d}/auxtriggers/0/channel'.format(self.id, awg), int(awg*4))
                    # set trigger threshold
                    self.daq.setDouble(
                        '/{:s}/triggers/in/{:d}/level'.format(self.id, int(awg*4)), 0.2)
            if grouping_index == 2:
                # set digital trigger
                self.daq.setInt(
                    '/{:s}/awgs/{:d}/auxtriggers/0/channel'.format(self.id, 0), 0)
                # set trigger threshold
                self.daq.setDouble(
                    '/{:s}/triggers/in/{:d}/level'.format(self.id, 0), 0.4)
            self.daq.sync()
        ## show current grouping info
        logger.info(
            '[%s] channel grouping: %s' %
            (self.id.upper(), grouping_name[self.grouping]))

    def update_pulse_length(self):
        for awg_index in range(4):
            hdinfo = self.daq.getList(
                '/{:s}/awgs/{:d}/waveform/waves/0'.format(self.id, awg_index))
            if len(hdinfo) == 0:
                self.waveform_length[awg_index] = -1
            elif len(hdinfo) == 1:
                length = int(
                    len(hdinfo[0][1][0]['vector'])/2)
                # consider two channel wave;
                # all of the ports keep the same length
                self.waveform_length[awg_index] = length
            else:
                raise Exception('Unknown HD infomation:\n', hdinfo)
        logger.info(
            '[%s-AWG] update_pulse_length: %r' %
            (self.id, self.waveform_length))

    def port_output(self,output=True,port=np.linspace(1,8,8)):
        # output = 1, open AWG port; output = 0, close AWG port;
        if isinstance(port, int):
            port = [port]
        for p in port:
            self.daq.setInt('/{:s}/sigouts/{:d}/on'.format(self.id,int(p-1)), int(output))

    @convertUnits(range_='V')
    def port_range(self,range_=None,port=np.linspace(1,8,8)):
        # AWG output range = 200mV,400mV,600mV,800mV,1V,2V,3V,4V,5V
        # range_=None -> return current range from ZI device
        if range_ == None:
            self.range = [None]*8
            for p in range(8):
                self.range[p] = round(self.daq.getDouble(
                                    '/{:s}/sigouts/{:d}/range'.format(self.id,int(p))),5)
            return self.range
        else:
            if isinstance(port, int):
                port = [port]
            for p in port:
                self.daq.setDouble(
                    '/{:s}/sigouts/{:d}/range'.format(self.id,int(p-1)),range_)

    @convertUnits(offset='V')
    def port_offset(self,offset=None,port=np.linspace(1,8,8)):
        ## set offset voltage to port; 
        ## offset = None --> return 'offset' info from ZI device;
        ## (minimum unit: 89.97uV in ZI device)
        if offset == None:
            self.offset = [None]*8
            for p in range(8):
                self.offset[p] = round(self.daq.getDouble(
                                    '/{:s}/sigouts/{:d}/offset'.format(self.id, p)),8)
        else:
            if isinstance(port, int):
                port = [port]
            for p in port:
                self.daq.setDouble(
                    '/{:s}/sigouts/{:d}/offset'.format(self.id,int(p-1)), offset)

    # -- bulid and send AWGs
    def _awg_builder(
        self, waveform: list, awg_index=0, loop=False):
        """ Build awg program for labone, then compile and send it to devices.
        """
        build_wave_num = 2**(self.grouping+1)
        wave_length = len(waveform[0])

        awg_program = get_HD_program(
            sample_rate=self.FS, number_port=build_wave_num,
            wave_length=wave_length, loop=loop)

        # complie index varies for different grouping
        awg_index_group = awg_index//(2**self.grouping)
        self._awg_upload_string(awg_program, awg_index=awg_index_group)
        self.update_pulse_length()

    def _awg_upload_string(self, awg_program, awg_index=0):
        """ write into waveforms sequencer and compile it.
        awg_program: waveforms sequencer text
        awg_index: this device's awgs sequencer index.
        If awgs grouping == 4*2, this index can be selected
        as 0,1,2,3.
        """
        awgModule = self.daq.awgModule()
        awgModule.set('awgModule/device', self.id)
        awgModule.set('awgModule/index', awg_index)  # AWG 0, 1, 2, 3
        awgModule.execute()
        awgModule.set('awgModule/compiler/sourcestring', awg_program)
        while awgModule.getInt('awgModule/compiler/status') == -1:
            time.sleep(0.1)
        # Ensure that compilation was successful
        if awgModule.getInt('awgModule/compiler/status') == 1:
            # compilation failed, raise an exception
            raise Exception(awgModule.getString(
                'awgModule/compiler/statusstring'))
        else:
            if awgModule.getInt('awgModule/compiler/status') == 0:
                logger.info(
                    "Compilation successful with no warnings,\
                    will upload the program to the instrument.")
            if awgModule.getInt('awgModule/compiler/status') == 2:
                logger.warning(
                    "Compilation successful with warnings,\
                    upload program to the instrument.")
                logger.warning("Compiler warning: ", awgModule.getString(
                    'awgModule/compiler/statusstring'))
            # wait for waveform upload to finish
            i = 0
            while awgModule.getDouble('awgModule/progress') < 1.0:
                time.sleep(0.1)
                i += 1
        logger.info('\n AWG upload successful. Output enabled. AWG Standby.')

    def _reload_waveform(self, waveform, awg_index=0, index=0):
        """ waveform: (numpy.array) one/two waves with unit amplitude.
            awg_index: this devices awg sequencer index
            index: this waveform index in total sequencer
        """
        waveform_native = convert_awg_waveform(waveform)
        logger.debug(
            '[%s-AWG%d] reload waveform length: %d' %
            (self.id, awg_index, len(waveform_native)))
        path = '/{:s}/awgs/{:d}/waveform/waves/{:d}'.format(
            self.id, awg_index, index)
        self.daq.setVector(path, waveform_native)

    def _update_when_error(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            iter_max = 2
            for i in range(iter_max):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    # complie index varies for different grouping
                    self.update_pulse_length()
            # return in the end for safety
            return func(self, *args, **kwargs)
        return wrapper

    @_update_when_error
    def send_waveform(self, waveform: list, awg_index=0):
        """
        Args:
            waveform (list): len(waveform)=2, for example, [[0],[0]].
            API of Zurich HD restrict this.
        Fill zeros at the end of waveform to match the prior waveform
        length or compile sequencer again, if the new wave is longer.
        """
        if len(waveform) != 2:
            raise ValueError("len(waveform) is not 2")
        _length_diff = self.waveform_length[awg_index] - len(waveform[0])
        if _length_diff < 0:
            _info_build = 'Bulid [%s-AWG%d] Sequencer2 (len=%r > %r)' % (
                self.id, awg_index, len(waveform[0]),
                self.waveform_length[awg_index])
            logger.info(_info_build)

            t0 = time.time()
            self._awg_builder(
                waveform=waveform,
                awg_index=awg_index)
            logger.info(
                '[%s-AWG%d] builder: %.3f s' %
                (self.id, awg_index, time.time()-t0))
            return
        else:
            waveform_add = [
                np.hstack((w, np.zeros(_length_diff))) for w in waveform
            ]
            self._reload_waveform(waveform_add, awg_index=awg_index)
            return



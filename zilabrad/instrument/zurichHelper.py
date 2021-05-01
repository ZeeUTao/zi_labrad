import textwrap  # to write sequencer's code
import time  # show total time in experiments
import enum

import numpy as np
from numpy import pi
from functools import wraps
import logging
import os
from collections import Iterable

from zilabrad.pyle.tools import singleton, convertUnits, Unit2SI
from zilabrad.instrument.QubitContext import qubitContext


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
hdawg8_grouping_awg_index = [[0, 1, 2, 3], [0, 2], [0]]


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

    def __init__(self, obj_name='qa_1', device_id='dev2592'):
        self.obj_name = obj_name
        self.id = device_id
        try:
            logger.info("\nBring up %s" % (self.id.upper()))
            self.daq = qubitContext().get_server('ziDAQ').daq
            self.connect()
            logger.info(self.daq)
            self.init_setup()
        except Exception as e:
            logger.error("Failed to initialize [%s]" % self.id.upper())
            raise e

    # -- set device status
    def connect(self):
        self.daq.connectDevice(self.id, '1gbe')

    def disconnect(self):
        self.daq.disconnectDevice(self.id)

    # def refresh_api(self):
    #     self.daq = qubitContext().get_server('ziDAQ').daq

    def init_setup(self):
        """ initialize device settings.
        """
        # AWG sample rate
        self.FS = 1.8e9
        self.daq.setInt('/{:s}/awgs/0/time'.format(self.id), 0)
        # clear all Registers value
        self.daq.setDouble('/{:s}/awgs/0/userregs/*'.format(self.id), 0)
        # use demodulator number in each sequence
        self.demodulator_number = 1
        # set result length,averages (default: no average)
        self.set_result_samples(1024, 1)
        self.qubit_frequency = []  # all demodulate frequency; unit: Hz
        self.paths = []  # save result path, equal to channel number
        # qa pulse length in AWGs; unit: sample number
        self.update_pulse_length()
        # qa demod time length; input unit: second
        self.set_demod_length(4096/self.FS)
        # qa result mode: integration --> return origin (I+iQ)
        self.set_qaSource_mode(qaSource.Integration.value)

        # set experimental parameters ##
        # set delay: all device trigger -> QA pulse start
        self.set_pulse_start(0)
        self.set_demod_start(0)  # set delay: QA pulse start -> QA demod start
        self.set_relaxation_length(relax_time=200e-6)  # unit: second

        # set integration mode: 0=standard mode, 1=spectroscopy mode;
        self.daq.setInt('/{:s}/qas/0/integration/mode'.format(self.id), 0)

        # set signal output (readin line)
        # awgs output mode: 0=plain, 1=modulation;
        self.daq.setInt('/{:s}/awgs/0/outputs/*/mode'.format(self.id), 0)
        # awgs hold maximum amplitude 1
        self.daq.setDouble(
            '/{:s}/awgs/0/outputs/*/amplitude'.format(self.id), 1)
        # close 50 Ohm output impedance
        self.daq.setInt('/{:s}/sigouts/*/imp50'.format(self.id), 0)
        self.awg_range = [1.5, 1.5]  # awg output range 1.5V(default);
        self.port_range(self.awg_range, port=[1, 2])

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

        # open AWGs port output
        self.port_output(output=True)
        # close Rerun
        self.daq.setInt('/{:s}/awgs/0/single'.format(self.id), 1)
        logger.info('%s: Complete Initialization' % self.id.upper())

    def sync_parameters(self, mode='get'):
        ''' Synchronize class instance and devices parameter
            Two mode: ['get','set']
                (get) update class instance value from 
                device by 'daq.get()'
                (set) upload class instance from PC to 
                device by 'daq.set()'
        '''
        if mode == 'get':
            self.result_samples = self.daq.getInt(
                '/{:s}/qas/0/result/length'.format(self.id))
            self.result_average_samples = self.daq.getInt(
                '/{:s}/qas/0/result/averages'.format(self.id))
            self.integration_length = self.daq.getInt(
                '/{:s}/qas/0/integration/length'.format(self.id))
            self.source = self.daq.getInt(
                '/{:s}/qas/0/result/source'.format(self.id))
            self.update_pulse_length()
            self.port_range(back=False)
        elif mode == 'set':
            self.set_result_samples()
            self.set_demod_length()
            self.set_qaSource_mode()
            self.port_range(self.awg_range)

    # -- set AWGs status
    def update_pulse_length(self):
        ''' Update self.waveform_length value via 
            daq.getList(), make sure send_waveform()
            having true length.
        '''
        qainfo = self.daq.getList(
            '/{:s}/awgs/0/waveform/waves/0'.format(self.id))
        if len(qainfo) == 0:
            self.waveform_length = -1
        elif len(qainfo) == 1:
            # qalen has double channel wave;
            self.waveform_length = int(len(qainfo[0][1][0]['vector'])/2)
        else:
            raise Exception('Unknown QA infomation:\n', qainfo)
        logger.info(
            '[%s] update_pulse_length: %r' % (self.id.upper(), self.waveform_length))

    def awg_run(self, awgs_index=0, _run=True):
        if _run:
            self.daq.syncSetInt('/{:s}/awgs/0/enable'.format(self.id), 1)
            logger.debug('\n AWG running. \n')
        else:  # Stop result unit
            self.daq.unsubscribe(self.paths)
            self.daq.setInt('/{:s}/qas/0/result/enable'.format(self.id), 0)

    def port_output(self, output=True, port=[1, 2]):
        # output = 1, open AWG port; output = 0, close AWG port;
        if isinstance(port, int):
            port = [port]
        for p in port:
            self.daq.setInt(
                '/{:s}/sigouts/{:d}/on'.format(self.id, int(p-1)), int(output))

    @convertUnits(_range=None)
    def port_range(self, _range=None, port=[1, 2], back=True):
        # AWG output range = 1.5V or 150mV
        # range_=None -> return current range from ZI device
        if isinstance(_range, type(None)):
            for p in [0, 1]:
                self.awg_range[p] = round(
                    self.daq.getDouble('/{:s}/sigouts/{:d}/range'
                                       .format(self.id, int(p))), 5)
            if back:
                return self.awg_range
        else:
            if isinstance(port, int):
                port = [port]
            if np.alen(_range) == 1:
                _range = [_range]*len(port)
            for r, p in zip(_range, port):
                p = p-1
                self.daq.setDouble('/{:s}/sigouts/{:d}/range'.format(
                    self.id, int(p)), r)
                time.sleep(0.05)
                # update current self.awg_range
                self.awg_range[p] = round(
                    self.daq.getDouble('/{:s}/sigouts/{:d}/range'
                                       .format(self.id, int(p))), 5)

    # -- send AWGs waveform
    def _awg_builder(self, number_port, wave_length, awg_index=0):
        """ Build waveforms sequencer. Then compile and send it to devices.
        """
        logger.info(
            'Bulid [%s-AWG0] Sequencer (len=%r > %r)' %
            (self.id.upper(), wave_length, self.waveform_length))
        if awg_index != 0:
            raise Exception('[%s] awg_index=%d out of range,here only one AWG !' % (
                self.id.upper(), awgs_index))
        t0 = time.time()
        # create default zeros waveform
        awg_program = get_QA_program_ManyDemod(
            sample_rate=int(self.FS),
            number_port=number_port,
            wave_length=wave_length,
            demod_number=self.demodulator_number)

        self._awg_upload_string(awg_program, awg_index=awg_index)
        self.update_pulse_length()  # updata self.waveform_lenght
        logger.info(
            '[%s-AWG0] builder: %.3f s' % (self.id.upper(), time.time()-t0))

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

    def _reload_waveform(self, waveform, awg_index=0, index=0):
        """ waveform: (numpy.array) one/two waves with unit amplitude.
            awg_index: this devices awg sequencer index
            index: this waveform index in total sequencer
        """
        waveform_native = convert_awg_waveform(waveform)
        path = '/{:s}/awgs/{:d}/waveform/waves/{:d}'.format(
            self.id, awg_index, index)
        self.daq.setVector(path, waveform_native)

    def send_waveform(self, waveform: list, awg_index=0, recursion=3):
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
                awg_index=awg_index)

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

    def clear_awg_sequence(self):
        for awg in [0]:
            self._awg_upload_string('//%s\nconst f_s = %s;' %
                                    (time.asctime(), self.FS), awg)
        print('clear [%s] all AWG sequencer' % self.id.upper())

    # -- set qa demod parameters
    @convertUnits(relax_time='s')
    def set_relaxation_length(self, relax_time):
        # send to device: Register 16
        self.daq.setDouble(
            '/{:s}/awgs/0/userregs/15'.format(self.id),
            int(relax_time*self.FS/8))

    def set_result_samples(self, length=None, averages=None):
        """ length*averages: repetition number
            length: result points number
            averages: average number for each point

            Meanwhile update repeat index in AWG sequencer
            and QA result length & averages parameter.
            ** averages must 2**n !!
        """
        if averages is not None:
            for k in range(20):
                if 2**k >= int(averages):
                    averages = 2**k
                    break
            self.result_average_samples = averages
        if length is not None:
            self.result_samples = length

        self.daq.setInt('/{:s}/qas/0/result/averages'.format(self.id),
                        self.result_average_samples)  # results averages
        self.daq.setInt('/{:s}/qas/0/result/length'.format(self.id),
                        self.result_samples*self.demodulator_number)  # results length
        # send to device: Register 1
        self.daq.setDouble(
            '/{:s}/awgs/0/userregs/0'.format(self.id),
            self.result_average_samples*self.result_samples)

    @convertUnits(pulse_start='s')
    def set_pulse_start(self, pulse_start):
        ''' demod_start: # All device trigger --> QA pulse start
            Here convert value from second to sample number, 
            8 samples as a unit.
        '''
        # unit: Sample Number
        # send to device: Register 2
        self.daq.setInt(
            '/{:s}/awgs/0/userregs/1'.format(self.id), int(pulse_start*self.FS/8))

    @convertUnits(demod_start='s')
    def set_demod_start(self, demod_start, demod_index=0):
        ''' demod_start: QA pulse start --> QA integration start
            Here convert value from second to sample number, 
            8 samples as a unit. 
            Avoiding giving 'Unit' class !

            If more than one demodulator in one sequence, 
            user should input a list to set each delay time 
            between different demodulators.
        '''
        if int(demod_index+1) > self.demodulator_number:
            logger.warning(
                'Must demod_index = %d < [%s].demodulator_number = %d )' %
                (demod_index, self.id.upper(), self.demodulator_number))
        wait_sample = int(demod_start*self.FS/8)  # unit: Sample Number
        # send to device: Register 3 ~ (self.demodulator_number+2)
        self.daq.setInt(
            '/{:s}/awgs/0/userregs/{:d}'.format(self.id, int(demod_index+2)), wait_sample)

    def set_demod_start_many(self, demod_start_list=[0]):
        ''' demod_start: QA pulse start --> QA integration start
            Here convert value from second to sample number, 
            8 samples as a unit. 
            Avoiding giving 'Unit' class !

            If more than one demodulator in one sequence, 
            user should input a list to set each delay time 
            between different demodulators.
        '''
        for demod_index in range(len(demod_start_list)):
            demod_start = Unit2SI(demod_start_list[demod_index])
            if int(demod_index+1) > self.demodulator_number:
                logger.warning(
                    'Must demod_index = %d < [%s].demodulator_number = %d )' %
                    (demod_index, self.id.upper(), self.demodulator_number))
            wait_sample = int(demod_start*self.FS/8)  # unit: Sample Number
            # send to device: Register 3 ~ (self.demodulator_number+2)
            self.daq.setInt(
                '/{:s}/awgs/0/userregs/{:d}'.format(self.id, int(demod_index+2)), wait_sample)

    @convertUnits(length='s')
    def set_demod_length(self, length=None):
        ''' length: set length for demodulate.
            INPUT: unit --> Second,
            SAVE: unit --> Sample number
            Demodulate has maximum length 4096.
            Ignore exceeding part.
        '''
        if length is None:
            sample_length = self.integration_length
        else:
            sample_length = int(length*self.FS)  # unit --> Sample Number

        if sample_length > 4096:
            logger.warning(
                'QA pulse lenght too long ( %.1f>%.1f ns)' %
                (length*1e9, 4096/(self.FS*1e-9)))
            sample_length = 4096  # set the maximum length
        self.daq.setDouble(
            '/{:s}/qas/0/integration/length'.format(self.id),
            sample_length)
        self.integration_length = self.daq.getDouble(
            '/{:s}/qas/0/integration/length'.format(self.id))

    @convertUnits(demod_number=None)
    def set_demodulator_number(self, demod_number=1):
        """ set demodulator number in QA AWGs sequence.
            default only one demodulator_0 
        """
        if demod_number != self.demodulator_number:
            self.demodulator_number = demod_number
            self.update_pulse_length()
            if self.waveform_length > 0:
                self._awg_builder(number_port=2,
                                  wave_length=self.waveform_length,
                                  awg_index=0)
            self.set_result_samples()
            logger.info(
                '[%s] set QA demodulator_number == %d' %
                (self.id.upper(), self.demodulator_number))

    # -- set qa demod mode
    @convertUnits(mode=None)
    def set_qaSource_mode(self, mode=None):
        if mode is None:
            mode = self.source
        if isinstance(mode, str):
            mode = qaSource[mode].value

        if mode == qaSource.Integration.value:
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
            '/{:s}/qas/0/result/source'.format(self.id), mode)
        # update 'self.source' info from ZI device
        self.source = self.daq.getInt(
            '/{:s}/qas/0/result/source'.format(self.id))
        logger.info(
            ' [%s] QA Source Mode: %s' % (self.id.upper(), qaSource.name.value[self.source]))

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
    def set_qubit_frequency(self, *frequency_list):
        # set integration weights, and result paths
        self.qubit_frequency = np.zeros(10)
        n_ch = len(list(frequency_list))
        if self.source == qaSource.Integration.value:
            if n_ch > 10:
                logger.warning('frequency list(len=%d) exceeds the max \
                channel number 10.' % n_ch)
            self.qubit_frequency[0:n_ch:1] = frequency_list[:10]
        if self.source == qaSource.Rotation.value:
            if n_ch > 5:
                logger.warning('frequency list(len=%d) exceeds the max \
                    channel number 5.' % n_ch)
            self.qubit_frequency[0:n_ch*2:2] = frequency_list[:5]
            self.qubit_frequency[1:n_ch*2:2] = frequency_list[:5]
        self._set_all_integration()  # set integration weights
        self._set_subscribe()  # set result paths

    def _set_all_integration(self):
        w_index = np.arange(0, 4096, 1)
        for channel, freq in enumerate(self.qubit_frequency):
            # assign real and image integration coefficient
            # integration settings for one I/Q pair
            w_real = np.cos(w_index*freq/self.FS*2*pi)
            w_imag = np.sin(w_index*freq/self.FS*2*pi)
            self.daq.setVector(
                '/{:s}/qas/0/integration/weights/{}/real'.format(
                    self.id, channel), w_real)
            self.daq.setVector(
                '/{:s}/qas/0/integration/weights/{}/imag'.format(
                    self.id, channel), w_imag)

    def _set_subscribe(self):
        """ set demodulate result parameters -> upload qa result's paths
        """
        # reset qa result
        self.daq.setInt('/{:s}/qas/0/result/reset'.format(self.id), 1)
        # start qa result module, wait value
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
            num_samples (int): expected number of samples, is equal to 
                               repeat_number*demodulator_number/averages_number
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
        num_samples = int(self.result_samples*self.demodulator_number)
        data = self._acquisition_poll(self.daq, self.paths,
                                      num_samples=num_samples, timeout=10)
        data_list = list(data.values())
        if self.source == 2:
            ks = range(int(len(data_list)/2))
            def get_doubleChannel(k): return data_list[2*k]+1j*data_list[2*k+1]
            data_list = list(map(get_doubleChannel, ks))
        return data_list


class zurich_hd(object):
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

    def __init__(self, obj_name='hd_1', device_id='dev8334'):
        self.id = device_id
        self.obj_name = obj_name
        try:
            logger.info('\nBring up %s' % (self.id.upper()))
            self.daq = qubitContext().get_server('ziDAQ').daq
            self.connect()
            logger.info(self.daq)
            self.init_setup()
        except Exception as e:
            logger.error("Failed to initialize [%s]" % self.id.upper())
            raise e

    # -- set device status
    def connect(self):
        self.daq.connectDevice(self.id, '1gbe')

    def disconnect(self):
        self.daq.disconnectDevice(self.id)

    # def refresh_api(self,labone_ip='localhost'):
    #     self.daq = qubitContext().get_server('ziDAQ').daq

    def init_setup(self):
        # sample rate
        self.daq.setDouble(
            '/{:s}/system/clocks/sampleclock/freq'.format(self.id), 2.4e9)
        self.FS = self.daq.getDouble(
            '/{:s}/system/clocks/sampleclock/freq'.format(self.id))
        # four awg's waveform length, unit --> Sample Number
        self.waveform_length = [0, 0, 0, 0]
        for awg_index in range(len(self.waveform_length)):
            # update current 'waveform_length' from ZI device
            self.update_pulse_length(awg_index=awg_index)
        self.port_output(output=True)  # open all signal output port
        self.awg_range = [None]*8
        self.port_range()      # update current range in device
        self.port_range([1]*8)  # set default output range: 1V
        self.offset = [None]*8
        self.port_offset()     # update current offset in device
        self.port_offset([0]*8)  # set default offset: 0

        self.awg_grouping(index=0)  # (default) index=0 --> 4x2 grouping mode
        # [AWG 0~3] digital trigger 1&2: slope --> rise
        self.daq.setInt('/{:s}/awgs/*/auxtriggers/*/slope'.format(self.id), 1)
        # (DIO) trigger in, use 50 Ohm impedance
        self.daq.setInt('/{:s}/triggers/in/*/imp50'.format(self.id), 1)
        # set trigger threshold, 0.1V as default level
        self.daq.setDouble('/{:s}/triggers/in/*/level'.format(self.id), 0.1)

        # use maximum sample rate in AWG sequence
        self.daq.setInt('/{:s}/awgs/*/time'.format(self.id), 0)
        # set ref clock mode as 'External'
        self.daq.setInt(
            '/{:s}/system/clocks/referenceclock/source'.format(self.id), 1)
        # awgs hold maximum amplitude 1
        self.daq.setDouble(
            '/{:s}/awgs/0/outputs/*/amplitude'.format(self.id), 1.0)
        # awgs output mode: 0=plain, 1=modulation;
        self.daq.setInt(
            '/{:s}/awgs/0/outputs/0/modulation/mode'.format(self.id), 0)

        self._unknown_settings()

        # close rerun
        self.daq.setInt('/{:s}/awgs/0/single'.format(self.id), 1)
        # Ensure that all settings have taken effect on the device before continuing.
        # self.daq.sync()
        logger.info('%s: Complete Initialization' % self.id.upper())
        return

    def _unknown_settings(self):
        # Unknown settings were suggested by ZI engineer
        # need set at same time by 'daq.set(list)'
        exp_setting = [
            ['/{:s}/awgs/0/dio/strobe/slope'.format(self.id), 0],
            ['/{:s}/awgs/0/dio/strobe/index'.format(self.id), 15],
            ['/{:s}/awgs/0/dio/valid/index'.format(self.id), 0],
            ['/{:s}/awgs/0/dio/valid/polarity'.format(self.id), 2],
            # 111 三位qubit results
            ['/{:s}/awgs/0/dio/mask/value'.format(self.id), 7],
            ['/{:s}/awgs/0/dio/mask/shift'.format(self.id), 1],
            ['/{:s}/raw/dios/0/extclk'.format(self.id), 2]]
        self.daq.set(exp_setting)
        return

    # -- set awg status
    def awg_run(self, awgs_index=[0, 1, 2, 3], _run=True):
        if isinstance(awgs_index, int):
            awgs_index = [awgs_index]
        for awg in awgs_index:
            # run:1, stop:0; specific AWG following awgs_index
            self.daq.setInt('/{:s}/awgs/{:d}/enable'
                            .format(self.id, awg), int(_run))
            logger.debug('[%s-AWG%d] awg_run status: %s.'
                         % (self.id.upper(), awg, str(_run)))

    def port_output(self, output=True, port=np.linspace(1, 8, 8)):
        # output = 1, open AWG port; output = 0, close AWG port;
        if isinstance(port, int):
            port = [port]
        for p in port:
            self.daq.setInt(
                '/{:s}/sigouts/{:d}/on'.format(self.id, int(p-1)), int(output))

    @convertUnits(_range=None)
    def port_range(self, _range=None, port=[1, 2, 3, 4, 5, 6, 7, 8], back=True):
        """ AWG output range -> -0.8=direct,200mV,400mV,600mV,800mV,1V,2V,3V,4V,5V
            _range = None -> return current range from ZI device
        """
        if isinstance(_range, type(None)):
            for p in range(8):
                direct_mode = self.daq.getInt('/{:s}/sigouts/{:d}/direct'
                                              .format(self.id, int(p)))
                if direct_mode:
                    self.awg_range[p] = -0.8  # is direct mode
                else:
                    self.awg_range[p] = round(
                        self.daq.getDouble('/{:s}/sigouts/{:d}/range'
                                           .format(self.id, int(p))), 5)
            if back:
                return self.awg_range
        else:
            if isinstance(port, int):
                port = [port]
            if np.alen(_range) == 1:
                _range = [_range]*len(port)
            for r, p in zip(_range, port):
                p = int(p-1)
                if r < 0:  # if direct mode
                    self.daq.setInt('/{:s}/sigouts/{:d}/direct'
                                    .format(self.id, p), 1)
                    self.awg_range[p] = -0.8
                else:  # if normal mode
                    if abs(self.awg_range[p] - r) > 2e-2:
                        self.daq.setDouble('/{:s}/sigouts/{:d}/range'.format(
                            self.id, p), r)
                        time.sleep(0.05)
                        # update current self.awg_range
                        self.awg_range[p] = round(
                            self.daq.getDouble('/{:s}/sigouts/{:d}/range'
                                               .format(self.id, p)), 5)

    @convertUnits(offset=None)
    def port_offset(self, offset=None, port=[1, 2, 3, 4, 5, 6, 7, 8]):
        """ set offset voltage to port; 
            offset = None --> return 'offset' info from ZI device;
            (minimum unit: 89.97uV in ZI device)
        """
        if isinstance(offset, type(None)):
            for p in range(8):
                self.offset[p] = round(self.daq.getDouble(
                    '/{:s}/sigouts/{:d}/offset'.format(self.id, p)), 8)
        else:
            if isinstance(port, int):
                port = [port]
            if np.alen(offset) == 1:
                offset = [offset]*len(port)
            for dc, p in zip(offset, port):
                p = int(p-1)
                if abs(dc-self.offset[p]) > 5e-5:
                    self.daq.setDouble(
                        '/{:s}/sigouts/{:d}/offset'.format(self.id, p), dc)
                    time.sleep(0.05)  # wait to sync
                    self.offset[p] = round(self.daq.getDouble(
                        '/{:s}/sigouts/{:d}/offset'.format(self.id, p)), 8)

    def awg_grouping(self, index=0):
        """ grouping_index:
                0 : 4x2 with HDAWG8; 2x2 with HDAWG4.
                1 : 2x4 with HDAWG8; 1x4 with HDAWG4.
                2 : 1x8 with HDAWG8.
            set AWG grouping mode, following path:
            '/dev_id/system/awg/channelgrouping', Configure
            how many independent sequencers, should run on
            the AWG and how the outputs are grouped by sequencer.
        """
        channelgrouping_name = ['4x2 with HDAWG8',
                                '2x4 with HDAWG8', '1x8 with HDAWG8']
        # update 'grouping' info
        self.grouping = self.daq.getInt(
            '/{:s}/system/awg/channelgrouping'.format(self.id))
        # try to set grouping mode
        if int(index) != int(self.grouping):
            self.daq.setInt(
                '/{:s}/system/awg/channelgrouping'.format(self.id), index)
            for awg in hdawg8_grouping_awg_index[index]:
                # set digital trigger input channel
                self.daq.setInt(
                    '/{:s}/awgs/{:d}/auxtriggers/0/channel'.format(self.id, awg), int(awg*2))
            self.daq.sync()
            self.grouping = self.daq.getInt(
                '/{:s}/system/awg/channelgrouping'.format(self.id))
        # show current grouping info
        logger.info(
            '[%s] channel grouping: %s' %
            (self.id.upper(), channelgrouping_name[self.grouping]))

    def update_pulse_length(self, awg_index):
        # for awg_index in range(4):
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
            '[%s-AWG%d] update_pulse_length: %r' %
            (self.id.upper(), awg_index, self.waveform_length))

    # -- bulid and send AWGs
    def _awg_builder(
            self, wave_length: int, awg_index=0, loop=False):
        """ Build awg program for labone, then compile and send it to devices.
        """
        build_wave_num = 2**(self.grouping+1)
        awg_program = get_HD_program(
            sample_rate=self.FS, number_port=build_wave_num,
            wave_length=wave_length, loop=loop)
        self._awg_upload_string(awg_program, awg_index=awg_index)
        self.update_pulse_length(awg_index=awg_index)

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
                    "\n  Compilation successful with no warnings,\
                    \n  will upload the program to the instrument.")
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
        logger.info(
            '\n AWG%d upload successful. Output enabled. AWG Standby.' % awg_index)

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

    def send_waveform(self, waveform: list, awg_index=0, recursion=3):
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
        wf_len = len(waveform[0])
        _n_ = self.waveform_length[awg_index] - wf_len
        if _n_ < 0:
            _info_build = 'Bulid [%s-AWG%d] Sequencer (len=%r>%r)' % (
                self.id.upper(), awg_index, wf_len,
                self.waveform_length[awg_index])
            logger.info(_info_build)
            t0 = time.time()
            self._awg_builder(
                wave_length=len(waveform[0]),
                awg_index=awg_index)
            logger.info(
                '[%s-AWG%d] builder: %.3f s' %
                (self.id.upper(), awg_index, time.time()-t0))
            self.send_waveform(
                waveform=waveform,
                awg_index=awg_index,
                recursion=recursion-1)
            return
        else:
            waveform_add = [np.hstack((wf, np.zeros(_n_))) for wf in waveform]
            try:
                self._reload_waveform(
                    waveform=waveform_add,
                    awg_index=awg_index)
            except Exception:
                self.update_pulse_length(awg_index=awg_index)
                self.send_waveform(
                    waveform=waveform,
                    awg_index=awg_index,
                    recursion=recursion-1)
            return

    def clear_awg_sequence(self):
        for awg in hdawg8_grouping_awg_index[self.grouping]:
            self._awg_upload_string('//%s\nconst f_s = %s;' %
                                    (time.asctime(), self.FS), awg)
        print('clear [%s] all AWG sequencer' % self.id.upper())


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
setTrigger(0b00);
repeat (getUserReg(0)) {  // = qa.result_samples * qa.result_average_samples
        // waitDigTrigger(1,1);
        setTrigger(0b11); // trigger output: rise
        wait(5); // trigger length: 22.2 ns / 40 samples
        setTrigger(0b00); // trigger output: fall
        wait(getUserReg(1)); // pulse wait time: all device trigger -> qa pulse start
        playWave($wave_play_string);
        wait(getUserReg(2)); // demod wait time: qa pulse start -> qa demod start
        setTrigger(AWG_INTEGRATION_ARM + AWG_INTEGRATION_TRIGGER + \
AWG_MONITOR_TRIGGER);// start demodulate
        setTrigger(AWG_INTEGRATION_ARM);// reset intergration
        waitWave();
        wait(getUserReg(3)); // wait relaxation (default 200us)
}
    """)
    awg_program = awg_program.replace(
        '$sample_rate', str(sample_rate))
    awg_program = awg_program.replace(
        '$wave_define_string', wave_define_string)
    awg_program = awg_program.replace(
        '$wave_play_string', wave_play_string)
    return awg_program


def get_QA_program_ManyDemod(
        sample_rate, number_port, wave_length, demod_number=1, *args, **kwargs):
    """awg program for labone
    Example:
    awg_program = get_QA_program_ManyDemod(sample_rate=int(1.8e9),
                                           number_port=2，
                                           demod_number=1)
    """
    wave_define_string = ""
    wave_play_string = ""
    for i in range(number_port):
        wave_define_string += "wave w" + str(i+1) +\
            "= zeros(" + str(wave_length) + ");\n"
        wave_play_string += (',' + str(i+1) + ',w'+str(i+1))
    wave_play_string = wave_play_string[1:]
    # delect the first comma

    demod_command_str = ''
    for k in range(demod_number):
        demod_command_str += """
        wait(getUserReg({:d})); // Registers {:d}: Demodulator_{:d} wait time
        setTrigger(AWG_INTEGRATION_ARM + AWG_INTEGRATION_TRIGGER + AWG_MONITOR_TRIGGER);// start demodulate
        setTrigger(AWG_INTEGRATION_ARM);// reset intergration""".format(int(k+2), int(k+3), k)

    awg_program = textwrap.dedent("""\
// $buliding_time
const f_s = $sample_rate;
$wave_define_string
setTrigger(AWG_INTEGRATION_ARM);// initialize integration
setTrigger(0b00);
repeat (getUserReg(0)) {  // = qa.result_samples * qa.result_average_samples
        // waitDigTrigger(1,1);
        setTrigger(0b11); // trigger output: rise
        // wait(5); // trigger length: 22.2 ns / 40 samples
        wait(getUserReg(1)); // Registers 2: pulse wait time
        playWave($wave_play_string);
        setTrigger(0b00); // trigger output: fall"""+demod_command_str+"""
        waitWave();
        wait(getUserReg(15)); // Registers 16: wait relaxation (default 200us)
}
    """)
    awg_program = awg_program.replace(
        '$buliding_time', time.asctime())
    awg_program = awg_program.replace(
        '$sample_rate', str(sample_rate))
    awg_program = awg_program.replace(
        '$wave_define_string', wave_define_string)
    awg_program = awg_program.replace(
        '$wave_play_string', wave_play_string)
    return awg_program


def get_HD_program(sample_rate: int, number_port: int, wave_length: int, loop=False):
    """ Return (str):
            awg program for labone
    """
    def raw_program(FS, define_str, play_str, loop):
        if loop:
            trigger_str = '//waitDigTrigger(1);'
        else:
            trigger_str = 'waitDigTrigger(1);'
        program = textwrap.dedent(f"""\
// {time.asctime()}
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
    Rotation = 2
    TRANS_STAT = 3
    CORR_TRANS = 4
    CORR_THRES = 5
    CORR_STAT = 6
    Integration = 7
    name = [
        'TRANS', 'THRES', 'Rotation', 'TRANS_STAT', 'CORR_TRANS',
        'CORR_THRES', 'CORR_STAT', 'Integration'
    ]

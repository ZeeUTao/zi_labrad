# Copyright (C) 2007  Matthew Neeley
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
### BEGIN NODE INFO
[info]
name = RS Source
version = 1.0
description = 

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""

from labrad.server import setting
from labrad.gpib import GPIBManagedServer, GPIBDeviceWrapper
from twisted.internet.defer import inlineCallbacks, returnValue

from labrad.units import Unit,Value

Hz,MHz,GHz = (Unit(s) for s in ['Hz','MHz','GHz'])

class RSWrapper(GPIBDeviceWrapper):
    @inlineCallbacks
    def initialize(self):
        self.frequency = yield self.getFrequency()
        self.amplitude = yield self.getAmplitude()
        self.outputStateKnown = False
        self.output = True

    @inlineCallbacks
    def getFrequency(self):
        self.frequency = yield self.query(':sour:freq?').addCallback(float)
        self.frequency = self.frequency*Hz
        returnValue(self.frequency)

    @inlineCallbacks
    def getAmplitude(self):
        self.amplitude = yield self.query(':sour:pow?').addCallback(float)
        self.amplitude = self.amplitude*Unit('dBm')
        returnValue(self.amplitude)

    @inlineCallbacks
    def setFrequency(self, f):
        if self.frequency != f:
            yield self.write(':sour:freq %f MHz' % f)
            self.frequency = f
    
    @inlineCallbacks
    def setAmplitude(self, a):
        if self.amplitude != a:
            yield self.write(':sour:pow %f' % a)
            self.amplitude = a

    @inlineCallbacks
    def setOutput(self, out):
        if self.output != out or not self.outputStateKnown:
            print 'self.output: ',self.output
            print 'self.outputStateKnown:',self.outputStateKnown
            yield self.write('outp %d' % int(out))
            self.output = out
            self.outputStateKnown = True
            
    # @inlineCallbacks
    # def setVideoPolarity(self,vidplo):
        # if vidplo == 'norm':
            # vidplo = yield self.write('pulm:outp:vid:pol \'%s\'' % vidplo)
        # elif vidplo == 'inv':
            # vidplo = yield self.write('pulm:outp:vid:pol \'%s\'' % vidplo) 

class RSServer(GPIBManagedServer):
    """Provides basic CW control for Anritsu MG3692C Microwave Generators"""
    # name = 'ANRITSU MG3692C' # Tongxing Yan, 2014-11-12.
    name = 'Anritsu Server'
    deviceName = 'ANRITSU MG3692C'
    deviceWrapper = RSWrapper

    @setting(10, 'Frequency', f=['v[MHz]'], returns=['v[MHz]'])
    def frequency(self, c, f=None):
        """Get or set the CW frequency."""
        dev = self.selectedDevice(c)
        if f is not None:
            yield dev.setFrequency(f)
        returnValue(dev.frequency)

    @setting(11, 'Amplitude', a=['v[dBm]'], returns=['v[dBm]'])
    def amplitude(self, c, a=None):
        """Get or set the CW amplitude."""
        dev = self.selectedDevice(c)
        if a is not None:
            yield dev.setAmplitude(a)
        returnValue(dev.amplitude)

    @setting(12, 'Output', os=['b'], returns=['b'])
    def output_state(self, c, os=None):
        """Get or set the output status."""
        dev = self.selectedDevice(c)
        if os is not None:
            yield dev.setOutput(os)
        returnValue(dev.output)
        
    @setting(13, 'VideoPolarity', vidpol=['s'], returns=['s'])
    def VideoPolarity(self, c, vidpol=None):
        """Set the polarity between modulating and modulated signal. You can enter 'norm' or 'inv'.
        'norm': the RF signal is suppressed during the pulse pause.
        'inv': the RF signal is suppressed during the pulse. """
        dev = self.selectedDevice(c)
        if vidpol is None:
            vidpol = yield dev.query('pulm:outp:vid:pol?')
            ans = vidpol
        elif vidpol == 'norm':
            yield dev.write("pulm:outp:vid:pol %s" % vidpol)
            ans = vidpol
        elif vidpol == 'inv':
            yield dev.write("pulm:outp:vid:pol %s" % vidpol)
            ans = vidpol
        returnValue(ans)
    
    @setting(14, 'Polarity', pol=['s'], returns=['s'])
    def Polarity(self, c, pol=None):
        """This command sets the polarity between modulating and modulated signal. This command is only effective for an external modulation signal.
        'norm': The FR signal is suppressed during the pulse pause.
        'inv': The Rf signal is suppressed during the pulse."""
        dev = self.selectedDevice(c)
        if pol is None:
            pol = yield dev.query('pulm:pol?')
            ans = pol
        elif pol == 'norm':
            yield dev.write("pulm:pol %s" % pol)
            ans = pol
        elif pol == 'inv':
            yield dev.write("pulm:pol %s" % pol)
            ans = pol
        returnValue(ans)
        
    @setting(15, 'TriggerLevel', TrigLev=['s'], returns=['s'])
    def TriggerLevel(self, c, TrigLev=None):
        """This command selects the external trigger level(threshold TTL,0.5V or -2.5V).
        'ttl': enter 'ttl'
        '0.5V': enter 'p0v5'
        '-2.5': enter 'm2v5'"""
        dev = self.selectedDevice(c)
        if TrigLev is None:
            Trig = yield dev.query('pulm:trig:ext:lev?')
            ans = Trig
            # if Trig == 'TTL':
                # ans = 'ttl'
            # returnValue(ans)
            # if Trig == 'P0V5':
                # ans = '0.5 V'
            # returnValue(ans)                
            # if Trig == 'm2V5':
                # ans = '-2.5 V'
            # returnValue(ans)
        elif TrigLev == 'ttl':
            yield dev.write("pulm:trig:ext:lev %s" % TrigLev)
            ans = TrigLev
        elif TrigLev == 'p0v5':
            yield dev.write("pulm:trig:ext:lev %s" % TrigLev)
            ans = '0.5 V' 
        elif TrigLev == 'm2v5':
            yield dev.write("pulm:trig:ext:lev %s" % TrigLev)
            ans = '-2.5 V'
        returnValue(ans)
        
    @setting(16, 'TrigExtImpednce', TrigExtImp=['s'], returns=['s'])
    def TrigExtImpednce(self, c, TrigExtImp=None):
        """This command selects the impedanc for external pulse trigger.
        'G50' means 50 ohm
        'G10K' means 10K ohm"""
        dev = self.selectedDevice(c)
        if TrigExtImp is None:
            TrigExtImp = yield dev.query('sour:pulm:trig:ext:imp?')
            ans = TrigExtImp
        elif TrigExtImp == 'G50':
            yield dev.write("sour:pulm:trig:ext:imp %s" % TrigExtImp)
            ans = '50 ohm'
        elif TrigExtImp == 'G10K':
            yield dev.write("sour:pulm:trig:ext:imp %s" % TrigExtImp)
            ans = '10k ohm' 
        returnValue(ans)
        
    @setting(17, 'PulmStat', state=['b'], returns=['b'])
    def PulmStat(self, c, state=None):
        """This command activates/deactivates the pulse modulation.
           e.g. pulmstat(False)
        """
        dev = self.selectedDevice(c)
        if state is None:
            ans = yield dev.query('pulm:stat?')
            state = bool(int(ans))
        else:
            yield dev.write('pulm:stat %d' % state)
        returnValue(state) 
    
    @setting(18, 'FrequencyMode', FreqMode=['s'], returns=['s'])
    def FrequencyMode(self, c, FreqMode=None):
        """This command sets the instrument operating mode. You can select 'cw/fixed/sweep/list'"""
        dev = self.selectedDevice(c)
        if FreqMode is not None:
            yield dev.write('freq:mode %s'%FreqMode)
        else:
            FreqMode = yield dev.query('freq:mode?')
        returnValue(FreqMode)
        
    @setting(19, 'FrequencyMultiplier', FreqMult=['v'], returns=['v'])
    def FrequencyMultiplier(self, c, FreqMult=None):
        """This command sets the multiplication factor of a subsequent downstream instrument.
        range: 1 to 10000"""
        dev = self.selectedDevice(c)
        if FreqMult is not None:
            yield dev.write('freq:mult %f'%FreqMult)
        else:
            FreqMult = yield dev.query('freq:mult?')
        returnValue(FreqMult)
        
    @setting(20, 'FrequencyOffset', FreqOffs=['s'], returns=['s'])
    def FrequencyOffset(self, c, FreqOffs=None):
        """This command sets the frequency offset of a downstream instrument.
        range: -67GHz to 67GHz"""
        dev = self.selectedDevice(c)
        if FreqOffs is not None:
            print FreqOffs
            yield dev.write('freq:offs %s'%FreqOffs)
        else:
            FreqOffs = yield dev.query('freq:offs?')
            # FreqOffs = float(ans)/1e9
        returnValue(FreqOffs)
        
    @setting(21, 'FrequencySpan', FreqSpan=['s'], returns=['s'])
    def FrequencySpan(self, c, FreqSpan=None):
        """This command specifies the span for the sweep.
        range: 100kHz to RFmax
        increment: 0.01Hz"""
        dev = self.selectedDevice(c)
        if FreqSpan is not None:
            yield dev.write('freq:span %s'%FreqSpan)
        else:
            FreqSpan = yield dev.query('freq:span?')
            # FreqSpan = float(ans)/1e6
        returnValue(FreqSpan)
        
    @setting(22, 'Frequencystart', FreqStar=['s'], returns=['s'])
    def Frequencystart(self, c, FreqStar=None):
        """This command sets the start frequency for the sweep mode.
        range: 2.5GHz to 23.5GHz"""
        dev = self.selectedDevice(c)
        if FreqStar is not None:
            print FreqStar
            yield dev.write("freq:star %s"%FreqStar)
        else:
            FreqStar = yield dev.query('freq:star?')
            # FreqStar = float(FreqStar)/1e6
        returnValue(FreqStar)
    
    @setting(23, 'FrequencyStop', FreqStop=['s'], returns=['s'])
    def FrequencyStop(self, c, FreqStop=None):
        """This command sets the start frequency for the sweep mode.
        range: 2.5GHz to 23.5GHz"""
        dev = self.selectedDevice(c)
        if FreqStop is not None:
            yield dev.write("freq:stop %s"%FreqStop)
        else:
            FreqStop = yield dev.query('freq:stop?')
            # FreqStop = float(ans)/1e6
        returnValue(FreqStop)
    
    @setting(24, 'FrequencyCenter', FreqCent=['s'], returns=['s'])
    def FrequencyCenter(self, c, FreqCent=None):
        """This command sets the center frequency of the sweep.
        range: 100kHz to RFmax
        increment: 0.01Hz"""
        dev = self.selectedDevice(c)
        if FreqCent is not None:
            yield dev.write("freq:cent %s"%FreqCent)
        else:
            FreqCent = yield dev.query('freq:cent?')
            # FreqCent = float(FreqCent)/1e6
        returnValue(FreqCent)
        
    @setting(25, 'FrequencyStep', FreqStep=['s'], returns=['s'])
    def FrequencyStep(self, c, FreqStep=None):
        """This command sets the step width for the frequency setting if ghe frequency values UP/DOWN are used and variation mode 'SOUR:FREQ:MODE USER' is selected.
        range: 1MHz to 10GHz
        increment: 0.01Hz"""
        dev = self.selectedDevice(c)
        if FreqStep is not None:
            yield dev.write('freq:step %s'%FreqStep)
        else:
            FreqStep = yield dev.query('freq:step?')
            # FreqSpan = float(ans)/1e6
        returnValue(FreqStep)
    
    @setting(26, 'SweepShape', SweepShape=['s'], returns=['s'])
    def SweepShape(self, c, SweepShape=None):
        """This command sets the cycle mode for a sweep sequence(shape).
        SAWtooth: One sweep runs from start to stop frequency. Each subsequent sweep starts at the start frequency, i.e. the shape of the sweep sequence resembles a sawtooth.
        TRIangle: One sweep runs from start to stop frequency and back,i.e. the shape of the sweep resembles a triangle. Each subsequent sweep starts at the start frequency."""
        dev = self.selectedDevice(c)
        if SweepShape is not None:
            yield dev.write('swe:shap %s'%SweepShape)  #can't write in, the screen doesn't change, but when query it changed.
        else:
            SweepShape = yield dev.query('swe:shap?')
        returnValue(SweepShape)
        
    @setting(27, 'SweepSpacing', SweepSpacing=['s'], returns=['s'])
    def SweepSpacing(self, c, SweepSpacing=None):
        """This command selects linear or logarithmic sweep spacing.
        i.e. SweepSpacing('LINear/LOGarithmic/ramp')"""
        dev = self.selectedDevice(c)
        if SweepSpacing is not None:
            yield dev.write('swe:spac %s'%SweepSpacing)
        else:
            SweepSpacing = yield dev.query('swe:spac?')
        returnValue(SweepSpacing)
        
    @setting(28, 'SweepStepLinear', SweStepLin=['s'], returns=['s'])
    def SweepStepLinear(self, c, SweStepLin=None):
        """This command sets the step width for the linear sweep.
        range: 0.001Hz to (stop - start)
        increment: 0.001Hz"""
        dev = self.selectedDevice(c)
        if SweStepLin is not None:
            yield dev.write('swe:step %s'%SweStepLin)
        else:
            SweStepLin = yield dev.query('swe:step?')
        returnValue(SweStepLin)
        
    @setting(29, 'SweepStepLog', SweStepLog=['s'], returns=['s'])
    def SweepStepLog(self, c, SweStepLog=None):
        """This command sets the step width for the logarithmic sweeps.
        range: 0.01PCT to 9999PCT
        increment: 0.01PCT"""
        dev = self.selectedDevice(c)
        if SweStepLog is not None:
            yield dev.write('swe:step:log %s pct'%SweStepLog)
        else:
            SweStepLog = yield dev.query('swe:step:log?')
        returnValue(SweStepLog)
    
    @setting(30, 'SweepDwell', SweepDwell=['s'], returns=['s'])
    def SweepDwell(self, c, SweepDwell=None):
        """This command sets the step width for the logarithmic sweeps.
        range: 1.0ms to 100s
        increment: 100.0e-6"""
        dev = self.selectedDevice(c)
        if SweepDwell is not None:
            yield dev.write('swe:dwel %s'%SweepDwell)
        else:
            SweepDwell = yield dev.query('swe:dwel?')
        returnValue(SweepDwell)
    
    @setting(31, 'Phase', Phase=['s'], returns=['s'])
    def Phase(self, c, Phase=None):
        """This command specifies the phase variation relative to the current phase. The variation can be specified in RADians
        range: -359.9 to 359.9
        increment: 0.1deg"""
        dev = self.selectedDevice(c)
        if Phase is not None:
            yield dev.write('phas %s'%Phase)
        else:
            Phase = yield dev.query('phas?')
        returnValue(Phase)
        
    # @setting(31, 'SweepMode', SweepMode=['s'], returns=['s'])
    # def SweepMode(self, c, SweepMode=None):
        # """This command sets the sweep mode, auto/single/step."""
        # dev = self.selectedDevice(c)
        # if SweepMode is not None:
            # yield dev.write('trig:sour %s'%SweepMode)
        # else:
            # SweepMode = yield dev.query('trig:sour?')
        # returnValue(SweepMode)
        
    @setting(32, 'RoscillatorSource', RoscSour=['s'], returns=['s'])
    def RoscillatorSource(self, c, RoscSour=None):
        """This command sets the sweep mode, auto/single/step."""
        dev = self.selectedDevice(c)
        if RoscSour is not None:
            yield dev.write('rosc:sour %s'%RoscSour)
        else:
            RoscSour = yield dev.query('rosc:sour?')
        returnValue(RoscSour)
    
    @setting(33, 'RFLevel', RFLevel=['s'], returns=['s'])
    def RFLevel(self, c, RFLevel=None):
        """This command sets the sweep mode, auto/single/step."""
        dev = self.selectedDevice(c)
        if RFLevel is not None:
            yield dev.write('pow:pow %s'%RFLevel)
        else:
            RFLevel = yield dev.query('pow:pow?')
        returnValue(RFLevel)
        
    @setting(34, 'RFLimit', RFLimit=['s'], returns=['s'])
    def RFLimit(self, c, RFLimit=None):
        """This command limits the maximum RF output level in CW and SWEEP mode."""
        dev = self.selectedDevice(c)
        if RFLimit is not None:
            yield dev.write('pow:lim %s'%RFLimit)
        else:
            RFLimit = yield dev.query('pow:lim?')
        returnValue(RFLimit)
    
    @setting(35, 'RFOffset', RFOffset=['s'], returns=['s'])
    def RFOffset(self, c, RFOffset=None):
        """This command specifise the constant level offset of a downstream atenuator/amplifier.
        range: -100dB to100dB"""
        dev = self.selectedDevice(c)
        if RFOffset is not None:
            yield dev.write('pow:offs %s'%RFOffset)
        else:
            RFOffset = yield dev.query('pow:offs?')
        returnValue(RFOffset)
        
    @setting(36, 'Resolution', Resolution=['s'], returns=['s'])
    def Resolution(self, c, Resolution=None):
        """This command selects the resolution for the level settings."""
        dev = self.selectedDevice(c)
        if Resolution is not None:
            yield dev.write('pow:res %s'%Resolution)
        else:
            Resolution = yield dev.query('pow:res?')
        returnValue(Resolution)
        
    @setting(37, 'RFStep', RFStep=['s'], returns=['s'])
    def RFStep(self, c, RFStep=None):
        """This command sets the step width for the level setting if UP and DOWN are used as the level values and variation mode is selected."""
        dev = self.selectedDevice(c)
        if RFStep is not None:
            yield dev.write('pow:step %s'%RFStep)
        else:
            RFStep = yield dev.query('pow:step?')
        returnValue(RFStep)
    
    @setting(38, 'RFStart', RFStart=['s'], returns=['s'])
    def RFStart(self, c, RFStart=None):
        """This command sets the RF start level in sweep mode."""
        dev = self.selectedDevice(c)
        if RFStart is not None:
            yield dev.write('pow:star %s'%RFStart)
        else:
            RFStart = yield dev.query('pow:star?')
        returnValue(RFStart)
        
    @setting(39, 'RFStop', RFStop=['s'], returns=['s'])
    def RFStop(self, c, RFStop=None):
        """This command sets the RF start level in sweep mode."""
        dev = self.selectedDevice(c)
        if RFStop is not None:
            yield dev.write('pow:stop %s'%RFStop)
        else:
            RFStop = yield dev.query('pow:stop?')
        returnValue(RFStop)
        
    @setting(40, 'RFShape', RFShape=['s'], returns=['s'])
    def RFShape(self, c, RFShape=None):
        """This command sets the cycle mode for a sweep sequence(shape).
        SAWTooth: One sweep runs from start level to stop level. The subsequent sweep starts at the start level, i.e. the shape of  sweep sequence resembles a sawtooth.
        TRIangle: One sweep runs from start to stop level and back,i.e. the shape of the sweep resembles a triangle. Each subsequent sweep starts at the start level again."""
        dev = self.selectedDevice(c)
        if RFShape is not None:
            yield dev.write('swe:pow:shap %s'%RFShape)
        else:
            RFShape = yield dev.query('swe:pow:shap?')
        returnValue(RFShape)
    
    @setting(41, 'RFSweepStep', RFSweStep=['s'], returns=['s'])
    def RFSweepStep(self, c, RFSweStep=None):
        """This command sets the step width for logarithmic sweeps.
        range: 0.01dB to 139dB
        increment: 0.01dB"""
        dev = self.selectedDevice(c)
        if RFSweStep is not None:
            yield dev.write('swe:pow:step %s'%RFSweStep)
        else:
            RFSweStep = yield dev.query('swe:pow:step?')
        returnValue(RFSweStep)
        
    @setting(42, 'RFSweepDwell', RFSweepDwell=['s'], returns=['s'])
    def RFSweepDwell(self, c, RFSweepDwell=None):
        """This command sets the time taken for each level step of the sweep.
        range: 1.0ms to 100s
        increment: 100.0e-6"""
        dev = self.selectedDevice(c)
        if RFSweepDwell is not None:
            yield dev.write('swe:pow:dwel %s'%RFSweepDwell)
        else:
            RFSweepDwell = yield dev.query('swe:pow:dwel?')
        returnValue(RFSweepDwell)
    
    @setting(43, 'TriggerSlope', TrigSlop=['s'], returns=['s'])
    def TriggerSlope(self, c, TrigSlop=None):
        """This command sets the slope of an externally applide trigger signal at the trigger input(BNC connector at the rear of the instrument).i.e. triggerslope(NEGative/POSitive)"""
        dev = self.selectedDevice(c)
        if TrigSlop is not None:
            yield dev.write('inp:trig:slop %s'%TrigSlop)
        else:
            TrigSlop = yield dev.query('inp:trig:slop?')
        returnValue(TrigSlop)
    
    @setting(44, 'ListMode', ListMode=['s'], returns=['s'])
    def ListMode(self, c, ListMode=None):  # still have problem
        """This command specifies how the list is to be processed. i.e.lismode(IMMediate(auto)/SINGle(step)/EXTernal(EXTernal))
        auto: each trigger event triggers a complete list cycle.
        step: each trigger event triggers only one step in the list processing cycle. The list is processed in ascending order."""
        dev = self.selectedDevice(c)
        if ListMode is None:
            ListMode = yield dev.query('list:trig:sour?')   
        elif ListMode =='step':
            yield dev.write('list:mode step')
        else:
            yield dev.write('list:trig:sour %s'%ListMode)
        returnValue(ListMode)
        
    @setting(45, 'ListDwell', ListDwell=['s'], returns=['s'])
    def ListDwell(self, c, ListDwell=None):
        """This command sets the time for which the instrument retains a setting.
        range: 0.5ms to 100s
        increment: 1e-4"""
        dev = self.selectedDevice(c)
        if ListDwell is not None:
            yield dev.write('list:dwel %s'%ListDwell)
        else:
            ListDwell = yield dev.query('list:dwel?')
        returnValue(ListDwell)
        
    @setting(46, 'ListStart', ListStart=['s'], returns=['s'])
    def ListStart(self, c, ListStart=None):
        """This command sets the start index of the index range which defines a subgroup of frequency/level value pairs in the current list. Only the values in the set index range are processed in list mode.
        range: 0 to list length"""
        dev = self.selectedDevice(c)
        if ListStart is not None:
            yield dev.write('list:ind:star %s'%ListStart)
        else:
            ListStart = yield dev.query('list:ind:star?')
        returnValue(ListStart)
    
    @setting(47, 'ListStop', ListStop=['s'], returns=['s'])
    def ListStop(self, c, ListStop=None):
        """This command sets the stop index of the index range which defines a subgroup of frequency/level value pairs in the current list. Only the values in the set index range are processed in list mode.
        range: 0 to list length"""
        dev = self.selectedDevice(c)
        if ListStop is not None:
            yield dev.write('list:ind:stop %s'%ListStop)
        else:
            ListStop = yield dev.query('list:ind:stop?')
        returnValue(ListStop)
    
    @setting(48, 'PulseModulationState', state=['b'], returns=['b'])
    def output(self, c, state=None):
        """Get or set the output state.
           e.g. PulseModulationState(False)
        """
        dev = self.selectedDevice(c)
        if state is None:
            ans = yield dev.query('pulm:stat?') 
            state = bool(int(ans))
        else:
            yield dev.write('pulm:stat %d' % state)
        returnValue(state)  
    
    # @setting(32, 'SweepPowerExecute', SwePowExec=['s'], returns=['s'])
    # def SweepPowerExecute(self, c, SwePowExec=None):
        # """This command sets the sweep mode, auto/manual/step."""
        # dev = self.selectedDevice(c)
        # if SwePowExec is None:
            # yield dev.write('swe:pow:exec')
        
__server__ = RSServer()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)

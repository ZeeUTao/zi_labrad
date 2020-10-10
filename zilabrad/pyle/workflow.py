# Copyright (C) 2010  Erik Lucero and Matthew Neeley
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


import os
from datetime import datetime

import labrad
from labrad.units import Unit
us, ns, V, mV, GHz, dBm = [Unit(s) for s in ('us', 'ns', 'V', 'mV', 'GHz', 'dBm')]


from zilabrad.pyle import registry


# default set of keys that will be set in the registry for each newly-created qubit
DEFAULT_KEYS = {
    'channels': [('timing', 'Preamp',   ['DR Lab Preamp 1', 'A']),
                 ('flux',   'FastBias', ['DR Lab FastBias 2', 'A']),
                 ('squid',  'FastBias', ['DR Lab FastBias 3', 'A']),
                 ('uwave',  'Iq',       ['DR Lab FPGA 1']),
                 ('meas',   'Analog',   ['DR Lab FPGA 7', 'A'])],

    'biasOperate':          0*V,
    'biasOperateSettling':  40*us,
    'biasReadout':          0*V,
    'biasReadoutSettling':  20*us,
    'biasReset':            [-1*V, -0.5*V, -1*V, -0.5*V],
    'biasResetSettling':    8*us,
    'biasStepEdge':         0*V,
    
    'f10':                  6*GHz,
    'f21':                  5.8*GHz,
    'fc':                   6.05*GHz,
    
    'measureAmp':           1,
    'measureAmp2':          0.5,
    'measureLenTop':        5*ns,
    'measureLenTop2':       5*ns,
    'measureLenFall':       20*ns,
    'measureLenFall2':      20*ns,
    
    'piAmp':                0.5, # amplitude of a pi pulse in XY
    'piAmpZ':               0.01, # amplitude of a pi pulse in Z
    'piFWHM':               6*ns,
    'piLen':                12*ns,
    
    'settlingAmplitudes':   [],
    'settlingRates':        [],
    
    'spectroscopyAmp':      0.01,
    'spectroscopyLen':      2000*ns,
    
    'squidBias':            0*V,
    'squidRampBegin':       0*V,
    'squidRampEnd':         2.5*V,
    'squidRampLength':      50*us,
    'squidReadoutDelay':    0*us,
    'squidReset':           0*V,
    'squidSwitchIntervals': [(25*us, 100*us)],
    
    'timingLagMeas':        0*ns,
    'timingLagUwave':       0*ns,

    'uwavePhase':           0.0,
    'uwavePower':           10*dBm,
}


def getYesNo(prompt, default=True):
    while True:
        answer = input(prompt).lower()
        if answer == '':
            return default
        elif answer == 'y':
            return True
        elif answer == 'n':
            return False
        else:
            print('Please enter "y" or "n"')


def getStringList(prompt, default=True):
    while True:
        try:
            answer = eval(input(prompt))
        except Exception:
            print('Please enter a list of quoted strings, e.g. ["a", "b"]')
        else:
            return answer


def createRegistryDirectories():
    with labrad.connect() as cxn:
        user = input('Enter user name: ')
        project = input('Enter project name: ')
        wafer = input('Enter wafer name (e.g. w100214A): ')
        die = input('Enter die name (e.g. r5c6): ')
        
        devices = []
        print('Enter device names one at a time (e.g. qubit0, res0, coupler0; blank line when done):')
        while True:
            device = input('  Enter device name (<ENTER> if done): ')
            if device == '':
                break
            devices.append(device)
        defaultKeys = getYesNo('Use default keys for devices? (Y/n) ')
        if not defaultKeys:
            while True:
                keyPath = getStringList('Enter path to directory with desired keys')
                try:
                    tempContext = cxn.context()
                    cxn.registry.cd(keyPath, context=tempContext)
                except Exception:
                    print('path %s does not exist in the registry!' % keyPath)
                else:
                    break
                
        
        session = input('Enter starting session (default = date only): ')
        today = datetime.now().strftime('%y%m%d')
        if session:
            session = '%s-%s' % (today, session)
        else:
            session = today
        
        print()
        print('The following registry structure will be created:')
        print('  ->', user)
        print('    ->', project)
        print('      ->', wafer)
        print('        ->', die)
        print('          ->', session)
        for device in devices:
            print('            ->', device)
            
        print()
        print('The following data vault structure will be created:')
        print('  ->', user)
        print('    ->', project)
        print('      ->', wafer)
        print('        ->', die)
        print('          ->', session)
            
        print()
        proceed = getYesNo('proceed? (Y/n) ', default=True)
        if proceed:
            print('making session folder...', end=' ')
            sessionPath = ['', user, project, wafer, die, session]
            reg = registry.RegistryWrapper(cxn, sessionPath)
            reg['config'] = devices
            print('done.')
            
            print('creating Data Vault directory...', end=' ')
            cxn.data_vault.cd(sessionPath, True)
            print('done.')
            
            for device in devices:
                if defaultKeys:
                    print('making folder for device %s...' % device, end=' ')
                    makeQubitRegistryDir(cxn, sessionPath, device)
                    print('done.')
                else:
                    print('copying keys for device %s from %s...' % (device, keyPath), end=' ')
                    reg[device] = registry.RegistryWrapper(cxn, keyPath)
                    print('done.')
            
            print('updating session variable...', end=' ')
            userPath = ['', user]
            reg = registry.RegistryWrapper(cxn, userPath)
            reg['sample'] = [project, wafer, die, session]
            print('done.')
        else:
            print('cancelled.')
        

def makeQubitRegistryDir(cxn, path, name):
    reg = registry.RegistryWrapper(cxn, path + [name])
    for key, val in list(DEFAULT_KEYS.items()):
        reg[key] = val


def switchSession(cxn, user, session=None, useDataVault=True):
    """Switch the current session."""
    userPath = ['', user]
    reg = registry.RegistryWrapper(cxn, userPath)
    print('Registry Root is', userPath)
    
    if session is None:
        samplePath = reg['sample']
        session = samplePath[-1]
        print('Sample Path is', samplePath)
    else:
        prefix, oldSession = reg['sample'][:-1], reg['sample'][-1]
        samplePath = prefix + [session]
        reg['sample'] = samplePath
        print('Sample Path changed to', samplePath)
    #Wrap only the last registry directory in samplePath
    for dir in samplePath[:-1]:
        reg = reg[dir]
    #Error if session doesn't exist at the end of the sample path
    if session is not None and session not in reg:
        print('Session "%s" not found.  Copying from "%s"...' % (session, oldSession), end=' ')
        reg[session] = reg[oldSession]
        print('Done.')
    reg = reg[session]
    
    # change data vault directory, creating new directories as needed
    if useDataVault:
        cxn.data_vault.cd(userPath + samplePath, True)
        print('Data Vault directory is', userPath + samplePath)
    else:
        print('WARNING: No data vault connection has been made. Set <useDataVault=True> to get a connection.')
    # change directory on disk
    #diskPath = os.path.join(*([USER_ROOT, user] + samplePath))
    #os.makedirs(diskPath)
    #os.chdir(diskPath)
    #print 'Current directory is', diskPath
    #global DATA_DIR
    #DATA_DIR = diskPath

    return reg


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


from zilabrad.pyle import registry


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
    # Wrap only the last registry directory in samplePath
    for dir in samplePath[:-1]:
        reg = reg[dir]
    # Error if session doesn't exist at the end of the sample path
    if session is not None and session not in reg:
        print('Session "%s" not found.  Copying from "%s"...'
              % (session, oldSession), end=' ')
        reg[session] = reg[oldSession]
        print('Done.')
    reg = reg[session]

    # change data vault directory, creating new directories as needed
    if useDataVault:
        cxn.data_vault.cd(userPath + samplePath, True)
        print('Data Vault directory is', userPath + samplePath)
    else:
        print('WARNING: No data vault connection has been made.\
Set <useDataVault=True> to get a connection.')
    return reg

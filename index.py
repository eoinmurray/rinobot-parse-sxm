import re
import os
import numpy as np

_end_tags = dict(grid=':HEADER_END:', scan='SCANIT_END', spec='[DATA]')


class NanonisFile(object):

    """
    Base class for Nanonis data files (grid, scan, point spectroscopy).

    Handles methods and parsing tasks common to all Nanonis files.

    Parameters
    ----------
    fname : str
        Name of Nanonis file.

    Attributes
    ----------
    datadir : str
        Directory path for Nanonis file.
    basename : str
        Just the filename, no path.
    fname : str
        Full path of Nanonis file.
    filetype : str
        filetype corresponding to filename extension.
    byte_offset : int
        Size of header in bytes.
    header_raw : str
        Unproccessed header information.
    """

    def __init__(self, fname):
        self.datadir, self.basename = os.path.split(fname)
        self.fname = fname
        self.filetype = self._determine_filetype()
        self.byte_offset = self.start_byte()
        self.header_raw = self.read_raw_header(self.byte_offset)

    def _determine_filetype(self):
        """
        Check last three characters for appropriate file extension,
        raise error if not.

        Returns
        -------
        str
            Filetype name associated with extension.

        Raises
        ------
        UnhandledFileError
            If last three characters of filename are not one of '3ds',
            'sxm', or 'dat'.
        """

        if self.fname[-3:] == '3ds':
            return 'grid'
        elif self.fname[-3:] == 'sxm':
            return 'scan'
        elif self.fname[-3:] == 'dat':
            return 'spec'
        else:
            raise UnhandledFileError('{} is not a supported filetype or does not exist'.format(self.basename))

    def read_raw_header(self, byte_offset):
        """
        Return header as a raw string.

        Everything before the end tag is considered to be part of the header.
        the parsing will be done later by subclass methods.

        Parameters
        ----------
        byte_offset : int
            Size of header in bytes. Read up to this point in file.

        Returns
        -------
        str
            Contents of filename up to byte_offset as a decoded binary
            string.
        """
        with open(self.fname, 'rb') as f:
            return f.read(byte_offset).decode()

    def start_byte(self):
        """
        Find first byte after end tag signalling end of header info.

        Caveat, I believe this is the first byte after the end of the
        line that the end tag is found on, not strictly the first byte
        directly after the end tag is found. For example in Scan
        __init__, byte_offset is incremented by 4 to account for a
        'start' byte that is not actual data.

        Returns
        -------
        int
            Size of header in bytes.
        """

        with open(self.fname, 'rb') as f:
            byte_offset = -1
            tag = _end_tags[self.filetype]
            line = f.readline()
            while(line != ''):
                if tag in line.strip().decode():
                    byte_offset = f.tell()
                    break

                line = f.readline()

            if byte_offset == -1:
                raise FileHeaderNotFoundError(
                        'Could not find the {} end tag in {}'.format(tag, self.basename)
                        )

        return byte_offset

class Scan(NanonisFile):

    """
    Nanonis scan file class.

    Contains data loading methods specific to Nanonis sxm files. The
    header is terminated by a 'SCANIT_END' tag followed by the \1A\04
    code. The NanonisFile header parse method doesn't account for this
    so the Scan __init__ method just adds 4 bytes to the byte_offset
    attribute so as to not include this as a datapoint.

    Data is structured a little differently from grid files, obviously.
    For each pixel in the scan, each channel is recorded forwards and
    backwards one after the other.

    Currently cannot take scans that do not have both directions
    recorded for each channel, nor incomplete scans.

    Parameters
    ----------
    fname : str
        Filename for scan file.

    Attributes
    ----------
    header : dict
        Parsed sxm header. Some fields are converted to float,
        otherwise most are string values.
    signals : dict
        Dict keys correspond to channel name, values correspond to
        another dict whose keys are simply forward and backward arrays
        for the scan image.

    Raises
    ------
    UnhandledFileError
        If fname does not have a '.sxm' extension.
    """

    def __init__(self, fname):
        _is_valid_file(fname, ext='sxm')
        super(self.__class__, self).__init__(fname)
        self.header = _parse_sxm_header(self.header_raw)

        # data begins with 4 byte code, add 4 bytes to offset instead
        self.byte_offset += 4

        # load data
        self.signals = self._load_data()

    def _load_data(self):
        """
        Read binary data for Nanonis sxm file.

        Returns
        -------
        dict
            Channel name keyed dict of each channel array.
        """
        channs = list(self.header['data_info']['Name'])

        nchanns = len(channs)
        nx, ny = self.header['scan_pixels']

        # assume both directions for now
        ndir = 2

        data_dict = dict()

        # open and seek to start of data
        f = open(self.fname, 'rb')
        f.seek(self.byte_offset)
        data_format = '>f4'
        scandata = np.fromfile(f, dtype=data_format)
        f.close()

        # reshape

        scandata_shaped = scandata.reshape(nchanns, ndir, nx, ny)

        # extract data for each channel
        for i, chann in enumerate(channs):
            chann_dict = dict(forward=scandata_shaped[i, 0, :, :],
                              backward=scandata_shaped[i, 1, :, :])
            data_dict[chann] = chann_dict

        return data_dict


class UnhandledFileError(Exception):

    """
    To be raised when unknown file extension is passed.
    """
    pass


class FileHeaderNotFoundError(Exception):

    """
    To be raised when no header information could be determined.
    """
    pass


def _parse_sxm_header(header_raw):
    """
    Parse raw header string.

    Empirically done based on Nanonis header structure. See Scan
    docstring or Nanonis help documentation for more details.

    Parameters
    ----------
    header_raw : str
        Raw header string from read_raw_header() method.

    Returns
    -------
    dict
        Channel name keyed dict of each channel array.
    """
    header_entries = header_raw.split('\n')
    header_entries = header_entries[:-3]

    header_dict = dict()
    entries_to_be_split = ['scan_offset',
                           'scan_pixels',
                           'scan_range',
                           'scan_time']

    entries_to_be_floated = ['scan_offset',
                             'scan_range',
                             'scan_time',
                             'bias',
                             'acq_time']

    entries_to_be_inted = ['scan_pixels']

    for i, entry in enumerate(header_entries):
        if entry == ':DATA_INFO:' or entry == ':Z-CONTROLLER:':
            count = 1
            for j in range(i+1, len(header_entries)):
                if header_entries[j].startswith(':'):
                    break
                if header_entries[j][0] == '\t':
                    count += 1
            header_dict[entry.strip(':').lower()] = _parse_scan_header_table(header_entries[i+1:i+count])
            continue
        if entry.startswith(':'):
            header_dict[entry.strip(':').lower()] = header_entries[i+1].strip()

    for key in entries_to_be_split:
        header_dict[key] = header_dict[key].split()

    for key in entries_to_be_floated:
        if isinstance(header_dict[key], list):
            header_dict[key] = np.asarray(header_dict[key], dtype=np.float)
        else:
            header_dict[key] = np.float(header_dict[key])
    for key in entries_to_be_inted:
        header_dict[key] = np.asarray(header_dict[key], dtype=np.int)

    return header_dict


def _parse_scan_header_table(table_list):
    """
    Parse scan file header entries whose values are tab-separated
    tables.
    """
    table_processed = []
    for row in table_list:
        # strip leading \t, split by \t
        table_processed.append(row.strip('\t').split('\t'))

    # column names are first row
    keys = table_processed[0]
    values = table_processed[1:]

    zip_vals = zip(*values)

    return dict(zip(keys, zip_vals))


def _is_valid_file(fname, ext):
    """
    Detect if invalid file is being initialized by class.
    """
    if fname[-3:] != ext:
        raise UnhandledFileError('{} is not a {} file'.format(fname, ext))


def print_to_asc(index, original_path, header):
    template = """:NANONIS_VERSION:
%s
:SCANIT_TYPE:
              FLOAT            MSBFIRST
:REC_DATE:
 %s
:REC_TIME:
%s
:REC_TEMP:
      %s
:ACQ_TIME:
      %s
:SCAN_PIXELS:
       %s       %s
:SCAN_FILE:
%s
:SCAN_TIME:
             %s             %s
:SCAN_RANGE:
           %s           %s
:SCAN_OFFSET:
             %s         %s
:SCAN_ANGLE:
            %s
:SCAN_DIR:
%s
:BIAS:
            %s
:DATA_INFO:
  Channel	Name	Unit	Direction	Calibration	Offset
  %s	%s	%s	%s	%s	%s

:SCANIT_END:


"""
    printable = template % (
        header["nanonis_version"],
        header["rec_date"],
        header["rec_time"],
        header["rec_temp"],
        header["acq_time"],
        header["scan_pixels"][0],
        header["scan_pixels"][1],
        header["scan_file"],
        header["scan_time"][0],
        header["scan_time"][1],
        header["scan_range"][0],
        header["scan_range"][1],
        header["scan_offset"][0],
        header["scan_offset"][1],
        header["scan_angle"],
        header["scan_dir"],
        header["bias"],
        header["data_info"]["Channel"][index],
        header["data_info"]["Name"][index],
        header["data_info"]["Unit"][index],
        header["data_info"]["Direction"][index],
        header["data_info"]["Calibration"][index],
        header["data_info"]["Offset"][index]
    )
    return printable


def process(path):
    p = re.compile(r'.*.sxm')

    if re.search(p, path):
        nf = Scan(path)

        for i, element in enumerate(nf.signals):
            printable = print_to_asc(i, path, nf.header)
            channel = nf.signals.keys()[i]

            shape = nf.signals[channel]['forward'].shape

            for direction in ['forward', 'backward']:

                if direction == 'forward':
                    dest = path + '[%s_fwd].asc' % channel
                    with open(dest, 'wt') as fp:
                        fp.write(printable)

                    with open(dest, 'a') as fp:
                        data_formatted = np.flipud(nf.signals[channel]['forward'].reshape((shape[1], shape[0])))
                        np.savetxt(fp, data_formatted)

                if direction == 'backward':
                    dest = path + '[%s_bwd].asc' % channel
                    with open(dest, 'wt') as fp:
                        fp.write(printable)

                    with open(dest, 'a') as fp:
                        data_formatted = np.flipud(nf.signals[channel]['backward'].reshape((shape[1], shape[0])))
                        np.savetxt(fp, data_formatted)


if __name__ == "__main__":
    import sys
    process(sys.argv[1])

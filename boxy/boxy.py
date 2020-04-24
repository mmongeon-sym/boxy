# https://support.box.com/hc/en-us/articles/360043697414-Using-Box-with-FTP-or-FTPS
# https://support.box.com/hc/en-us/articles/360044194853-I-m-Having-Trouble-Using-FTP-with-Box

from io import BytesIO

import dataclasses
import re

import os
import pathlib
import ftplib
import datetime
import shutil
import time

import functools
import sys
from types import ModuleType, FunctionType
from gc import get_referents
import logging
import logging.config
import hashlib
import ssl
import tempfile
import inspect

def rename_dataclass(obj , name: str = None):
    if name is None:
        return obj

    obj.__doc__ = (obj.__class__.__name__ +
                       str(inspect.signature(obj.__class__)).replace(' -> None', ''))
    obj.__class__.__name__ = name
    obj.__class__.__qualname__ = name
    return obj

@functools.lru_cache(maxsize=1024, typed=True)
def is_databricks_env():
    if os.environ.get('DATABRICKS_RUNTIME_VERSION') is not None:
        return True
    else:
        return False


@functools.lru_cache(maxsize=1024, typed=True)
def databricks_version():
    if is_databricks_env() is True:
        return float(os.environ.get('DATABRICKS_RUNTIME_VERSION'))

    else:
        return None


@functools.lru_cache(maxsize=1024, typed=True)
def is_pyspark():
    try:
        import pyspark
        return True
    except ModuleNotFoundError as e:
        return False


@functools.lru_cache(maxsize=1024, typed=True)
def gettempdir():
    # https://docs.databricks.com/data/databricks-file-system.html#local-file-apis-for-deep-learning
    # https://docs.databricks.com/applications/deep-learning/data-prep/ddl-storage.html#prepare-storage-for-data-loading-and-model-checkpointing
    if is_databricks_env() is True:
        if databricks_version() >= 6.0:
            return '/dbfs/tmp/'
        elif databricks_version() in (5.5, 5.4):
            return '/dbfs/ml'
        else:
            return tempfile.gettempdir()
    else:
        return tempfile.gettempdir()


@functools.lru_cache(maxsize=1024, typed=True)
def SHA1_file(path: str, mtime: datetime.datetime, bytes: int):
    if mtime is None:
        raise AttributeError("mtime must be provided")
    if bytes is None:
        raise AttributeError("bytes must be provided")
    h = hashlib.sha1()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    if os.path.exists(path) is False:
        raise FileNotFoundError(path)
    with open(path, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return str(h.hexdigest()).upper()


@functools.lru_cache(maxsize=1024, typed=True)
def MD5_file(path: str, mtime: datetime.datetime, bytes: int):
    if mtime is None:
        raise AttributeError("mtime must be provided")
    if bytes is None:
        raise AttributeError("bytes must be provided")

    h = hashlib.md5()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    if os.path.exists(path) is False:
        raise FileNotFoundError(path)
    with open(path, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return str(h.hexdigest()).upper()


def disable_java_logging():
    import logging

    logging.getLogger("java_gateway").setLevel(logging.ERROR)
    logging.getLogger("java_gateway.run").setLevel(logging.ERROR)
    # Disable noisy Databricks JVM
    try:
        import spark
        logger = spark._jvm.org.apache.log4j
        logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

    except Exception as e:
        pass
    # Disable noisy Databricks JVM
    # https://forums.databricks.com/questions/17799/-infopy4jjava-gatewayreceived-command-c-on-object.html
    try:
        logging.getLogger("py4j").setLevel(logging.ERROR)
    except Exception as e:
        pass


disable_java_logging()


def get_logger(name=__name__):
    log_rd = {
            'version':                  1,
            'disable_existing_loggers': False,
            'formatters':               {
                    'standard': {

                            'format':  '%(asctime)s[%(module)s.%(funcName)s][%(levelname)s] %(message)s'.format(
                                    ),
                            'datefmt': '%Y-%m-%dT%I:%M:%S%z'
                            }
                    },
            'handlers':                 {
                    'default': {
                            'level':     'INFO',
                            'formatter': 'standard',
                            'class':     'logging.StreamHandler',
                            },

                    },
            'loggers':                  {
                    '': {
                            'handlers': ['default',
                                         # 'rotate_file'
                                         ],
                            'level':    'INFO',
                            },
                    }
            }

    logging.config.dictConfig(log_rd)
    return logging.getLogger(name)


def getsizeof(obj):
    BLACKLIST = (type, ModuleType, FunctionType)
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: ' + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


@functools.lru_cache(maxsize=1024, typed=True)
def humansize(nbytes: int):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    if nbytes == 0:
        return '0 B'
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])


@dataclasses.dataclass(init=True, repr=False, eq=True, order=True, unsafe_hash=False, frozen=False)
class Timer(object):
    start_dt: datetime.datetime = dataclasses.field(init=False, default=None)
    end_dt: datetime.datetime = dataclasses.field(init=False, default=None)
    duration: (datetime.timedelta, None) = dataclasses.field(init=False, default=None)
    running: bool = dataclasses.field(init=False, default=None)
    finished: bool = dataclasses.field(init=False, default=None)
    tz: datetime.timezone = dataclasses.field(init=False,
                                              default=datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo)
    init_dt: datetime.datetime = dataclasses.field(init=False, default=None)
    format: str = dataclasses.field(init=False, default='%Y%m%dT%H:%M:%S.%f%z')
    items: int = dataclasses.field(init=False, default=None)
    bytes: int = dataclasses.field(init=False, default=None)
    item_name: str = dataclasses.field(init=False, default='items')

    def __repr__(self):
        if self.status == 'STARTED':
            return f"Timer(started='{self.start_dt.isoformat(sep='T')}', duration='{(self.now_dt - self.start_dt)}')"
        elif self.status == 'INIT':
            return f"Timer(initialized='{self.init_dt.isoformat(sep='T')}')"
        elif self.status == 'FINISHED':
            msg = [f"duration='{self.duration_str}'"]
            if self.bytes is not None:
                msg.append(f"size='{self.bytes_str}'")
                msg.append(f"speed='{self.bytes_per_second_str}'")
            if self.items is not None:
                msg.append(f"iops='{self.items_per_second_str}'")
            msg_str = ", ".join(msg)
            return f"Timer(finished='{self.end_dt.isoformat(sep='T')}', {msg_str})"
        else:
            return f"Timer(status='{self.status}', '{self.duration_str})"

    @property
    def status(self):
        if self.running is None and self.finished is None:
            return "INITIALIZED"
        if self.running is True and self.finished is not True:
            return "STARTED"
        if self.running is False and self.finished is True:
            return "FINISHED"

    @property
    def now_dt(self):
        return datetime.datetime.now(tz=self.tz)

    def start(self):
        self.start_dt = self.now_dt
        self.running = True
        self.finished = False

    def __post_init__(self):
        self.init_dt = datetime.datetime.now(tz=self.tz)

    def stop(self,
             n_items: int = None,
             n_bytes: int = None,
             item_name: str = None
             ):
        if n_items is not None and isinstance(n_items, (int, float)):
            self.items = n_items
        if n_bytes is not None and isinstance(n_bytes, int):
            self.bytes = n_bytes
        if item_name is not None and isinstance(item_name, str):
            self.item_name = item_name
        self.running = False
        self.finished = True
        if self.end_dt is None:
            self.end_dt = datetime.datetime.now(tz=self.tz)
        if self.duration is None:
            self.duration = self.end_dt - self.start_dt

    @property
    def bytes_str(self):
        if self.bytes is not None:
            return humansize(self.bytes)

    @property
    def items_per_second(self):
        if self.items is not None and self.duration is not None:
            return round(float((self.items / float(self.duration_seconds))), 1)

    @property
    def items_per_second_str(self):
        if self.items is not None and self.duration is not None:
            return str(self.items_per_second) + " " + self.item_name + "/s"

    @property
    def bytes_per_second(self):
        if self.bytes is not None and self.duration is not None:
            return round(float(self.bytes) / float(self.duration_seconds), 1)

    @property
    def bytes_per_second_str(self):
        if self.bytes is not None and self.duration is not None:
            return humansize(self.bytes_per_second) + "/s"

    @property
    def megabytes_per_second(self):
        if self.bytes is not None and self.duration is not None:
            return round(float(self.bytes) / float(self.duration_seconds) / 1024 / 1024, 1)

    @property
    def megabytes_per_second_str(self):
        if self.bytes is not None and self.duration is not None:
            return str(self.megabytes_per_second) + " MB/s"

    @property
    def bits_per_second(self):
        if self.bytes is not None and self.duration is not None:
            return round(float(self.bytes * 8) / float(self.duration_seconds), 1)

    @property
    def megabits_per_second(self):
        if self.bytes is not None and self.duration is not None:
            return round(float(self.bytes * 8 / 1048576) / float(self.duration_seconds), 1)

    @property
    def megabits_per_second_str(self):
        if self.bytes is not None and self.duration is not None:
            return str(self.megabits_per_second) + " Mbit/s"

    @property
    def duration_seconds(self):
        if self.duration is not None:
            return float(self.duration.total_seconds())
        else:
            return None

    @property
    def duration_str(self):
        if self.duration is not None:
            return str(self.duration)[:-3]

    def strftime(self, object: datetime.datetime, format: (str, None) = None):
        if format is None:
            format = self.format
        return object.strftime(fmt=format)

    def humansize(self, bytes: int):
        if bytes is not None and isinstance(bytes, int) or isinstance(object, float):
            suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
            if bytes == 0:
                return '0 B'
            i = 0
            while bytes >= 1024 and i < len(suffixes) - 1:
                bytes /= 1024.
                i += 1
            f = ('%.2f' % bytes).rstrip('0').rstrip('.')
            return '%s %s' % (f, suffixes[i])

    def __enter__(self):
        """

        Automatically start the timer when used as a context manager.
        Executes self.start() when entered

        Example:
            with Timer(action='Wrote', item_name='lines', log=True) as t:
                .....

        Returns:
            self (:obj:`Timer`): Returns it self as a context manager.

        """
        if hasattr(self, 'running') and self.running is True:
            pass
        else:
            self.start()
            return self

    def __exit__(self, *args):
        """
        Automatically stop the timer when used as a context manager.
        Executes self.stop() when exited

        Returns:
            :obj:`None`: :obj:`None`

        """

        self.stop()

    def sleep(self, seconds: int):
        """
        Artificially add processing time to a Timer by executing time.sleep for N seconds.

        Parameters:
            seconds (:obj:`int`): Number of seconds to sleep using time.sleep

        Returns:
            :obj:`None`: :obj:`None`
        """
        time.sleep(seconds)


@functools.lru_cache(maxsize=1024, typed=True)
def parse_time(time_str: str):
    # ex parse_time('19700131000000.000')
    #    returns: datetime.datetime(1970, 1, 30, 17, 0, tzinfo=datetime.timezone(datetime.timedelta(days=-1,
    #    seconds=61200), 'Pacific Daylight Time'))

    return datetime.datetime.strptime(time_str + '000' + '+0000', '%Y%m%d%H%M%S.%f%z').astimezone(
            datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo)


@functools.lru_cache(maxsize=1024, typed=True)
def datetime_to_ftptime(dt: datetime.datetime):
    # ex parse_time('19700131000000.000')
    #    returns: datetime.datetime(1970, 1, 30, 17, 0, tzinfo=datetime.timezone(datetime.timedelta(days=-1,
    #    seconds=61200), 'Pacific Daylight Time'))
    dt_noz = dt.astimezone(datetime.timezone.utc)

    return dt_noz.strftime('%Y%m%d%H%M%S') + "." + dt_noz.strftime('%f')[0:3]


@functools.lru_cache(maxsize=1024, typed=True)
def parse_unix_ts(unix_time: str):
    # ex parse_time('1586345134')
    #    returns: datetime.datetime(1970, 1, 30, 17, 0, tzinfo=datetime.timezone(datetime.timedelta(days=-1,
    #    seconds=61200), 'Pacific Daylight Time'))

    return datetime.datetime.fromtimestamp(float(unix_time)).replace(tzinfo=datetime.timezone.utc).astimezone(
            datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo)


class FTPFileNotFoundError(Exception):
    def __init__(self, path, classname=None, errors=None):
        super().__init__(path)
        self.__class__.__name__ = classname
        self.__class__.__qualname__ = classname
        self.errors = errors

class FTPFileExistsError(Exception):
    def __init__(self, path, classname=None, errors=None):
        super().__init__(path)
        self.__class__.__name__ = classname
        self.__class__.__qualname__ = classname
        self.errors = errors

class BoxFileExistsError(Exception):
    def __init__(self, path, classname=None, errors=None):
        super().__init__(path)
        self.__class__.__name__ = classname
        self.__class__.__qualname__ = classname
        self.errors = errors

class FTPFolderExistsError(Exception):
    def __init__(self, path, classname=None, errors=None):
        super().__init__(path)
        self.__class__.__name__ = classname
        self.__class__.__qualname__ = classname
        self.errors = errors



class FTPFolderNotFoundError(Exception):
    def __init__(self, path, classname=None, errors=None):
        super().__init__(path)
        self.__class__.__name__ = classname
        self.__class__.__qualname__ = classname
        self.errors = errors


@dataclasses.dataclass(init=True, repr=False, eq=True, order=True, unsafe_hash=False, frozen=False)
class LocalFile(object):
    path: str = dataclasses.field(init=True)
    directory: str = dataclasses.field(init=False)
    name: str = dataclasses.field(init=False)
    exists: bool = dataclasses.field(init=False)
    size: str = dataclasses.field(init=False)
    bytes: int = dataclasses.field(init=False)
    extension: str = dataclasses.field(init=False)
    ctime: datetime.datetime = dataclasses.field(init=False)
    mtime: datetime.datetime = dataclasses.field(init=False)
    _md5: str = dataclasses.field(init=False, default=None)

    log = get_logger()

    def __repr__(self,
                 size=True,
                 mtime=False,
                 ctime=False,
                 exists=True
                 ):
        if self.exists is True:
            msgs = []
            if exists is True and self.exists is not None:
                msgs.append(f"exists={self.exists}")
            if size is True and self.size is not None:
                msgs.append(f"size='{self.size}'")
            if mtime is True and self.mtime is not None:
                msgs.append(f"mtime='{self.mtime}'")
            if ctime is True and self.ctime is not None:
                msgs.append(f"ctime='{self.ctime}'")

            if len(msgs) > 0:
                msgs_str = ", ".join(msgs)
                return f"LocalFile('{self.path}', {msgs_str})"
            else:
                return f"LocalFile('{self.path}')"
        else:
            return f"{self.__class__.__name__}('{self.path}', exists=False)"

    def __str__(self,
                size=True,
                mtime=False,
                ctime=False,
                exists=True
                ):
        return self.__repr__(
                size=True,
                mtime=False,
                ctime=False,
                exists=True
                )

    def __post_init__(self):
        self.directory = os.path.dirname(self.path)
        self.name = pathlib.Path(self.path).name
        self.extension = pathlib.Path(self.path).suffix
        if os.path.exists(self.path) is True:
            stat = os.stat(self.path)
            self.exists = True
            self.size = humansize(stat.st_size)
            self.bytes = stat.st_size
            if hasattr(stat, 'st_ctime'):
                self.ctime = parse_unix_ts(getattr(stat, 'st_ctime'))
            if hasattr(stat, 'st_birthtime') is True and getattr(stat, 'st_birthtime') is not None:
                self.ctime = parse_unix_ts(getattr(stat, 'st_birthtime'))
            self.mtime = parse_unix_ts(stat.st_mtime)
        else:
            self.exists = False
            self.size = humansize(0)
            self.bytes = 0
            self.ctime = None
            self.mtime = None

    @property
    def md5(self):
        self.assert_exists()
        if self._md5 is not None:
            return self._md5
        else:
            self._md5 = MD5_file(path=self.path, mtime=self.mtime, bytes=self.bytes)
            return self._md5

    def refresh(self):
        self.__post_init__()

    def clear(self):
        self.exists = False
        self.size = humansize(0)
        self.bytes = 0
        self.ctime = parse_time('19700131000000.000')
        self.mtime = parse_time('19700131000000.000')

    def makedirs(self):
        if os.path.exists(self.directory) is False:
            os.makedirs(self.directory, exist_ok=True)

    def assert_exists(self):
        if self.exists is False:
            raise FileNotFoundError(self.path)

    def assert_not_exists(self):
        if self.exists is True:
            raise FileExistsError(self.path)

    def remove(self):
        if os.path.exists(self.path) is True:
            os.remove(self.path)
            if os.path.exists(self.path) is False:
                self.log.info(f"Removed: {self.__repr__(exists=False, size=True)}")
                self.refresh()
            else:
                self.log.error(f"Failed to remove: {self.__repr__(exists=True, size=True)}")
                raise FTPFileExistsError

    def read(self):
        self.assert_exists()
        with open(self.path, mode='r') as infile:
            string = infile.read()
        self.log.info(f"Read {len(string.splitlines())} lines from: {self.__repr__(exists=False, size=True)}")
        return string

    def readlines(self):
        string = self.read()
        return string.splitlines()

    def write(self, string):
        lines = str(string).splitlines()
        self.makedirs()
        with Timer() as t:
            with open(self.path, mode='w') as outfile:
                outfile.writelines(lines)
        self.refresh()
        t.stop(n_bytes=self.bytes, n_items=len(lines), item_name='lines')
        self.log.info(f"Wrote {len(lines)} lines,"
                      f" {len(string)} chars "
                      f"to: {self.__repr__(exists=False, size=True)} "
                      f"in '{t.duration_str}', '{t.bytes_per_second_str}', '{t.items_per_second_str}'")

    def writelines(self, lines):
        self.makedirs()
        with Timer() as t:
            with open(self.path, mode='w') as outfile:
                outfile.writelines(lines)
        self.refresh()
        t.stop(n_bytes=self.bytes, n_items=len(lines), item_name='lines')
        self.log.info(f"Wrote {len(lines)} lines "
                      f"to: {self.__repr__(exists=False, size=True)} "
                      f"in '{t.duration_str}', '{t.bytes_per_second_str}', '{t.items_per_second_str}'")

    def rename(self, name):
        new_path = os.path.join(self.directory, name)
        shutil.move(self.path, os.path.join(self.directory, name), copy_function=shutil.copy2)
        new_file = LocalFile(path=new_path)
        if new_file.exists is True:
            return self.log.info(f"Renamed {self.__repr__(exists=False, size=False)} "
                                 f"to {new_file.__repr__(exists=False, size=False)}")
        else:
            self.log.error(f"Failed to rename {self.__repr__(exists=False, size=False)} "
                           f"to {new_file.__repr__(exists=False, size=False)}")
            raise FileNotFoundError(new_path)

    def move(self, new_path):
        parts = pathlib.Path(new_path)
        directory = parts.parent
        filename = parts.name

        shutil.move(self.path, os.path.join(directory, filename), copy_function=shutil.copy2)
        new_file = LocalFile(path=new_path)
        if new_file.exists is True:
            return self.log.info(f"Moved {self.__repr__(exists=False, size=False)} "
                                 f"to {new_file.__repr__(exists=False, size=False)}")
        else:
            self.log.error(f"Failed to move {self.__repr__(exists=False, size=False)} "
                           f"to {new_file.__repr__(exists=False, size=False)}")
            raise FileNotFoundError(new_path)

    def copy(self, path):
        parts = pathlib.Path(path)
        if os.path.exists(path) is True:
            os.remove(path)
        os.makedirs(parts.parent)
        shutil.copy2(self.path, path)
        new_file = LocalFile(path=path)
        if new_file.exists is True:
            return self.log.info(f"Copied {self.__repr__(exists=True, size=True)} "
                                 f"to {new_file.__repr__(exists=True)}")
        else:
            self.log.info(f"Failed to copy {self.__repr__(exists=True, size=True)} "
                          f"'to {new_file.__repr__(size=False, exists=False)}")

    def set_mtime(self, dt: datetime.datetime):
        t = dt.astimezone(tz=datetime.timezone.utc).timestamp()
        os.utime(self.path, times=(t, t))
        self.mtime = dt
        return self

    def set_ctime(self, dt: datetime.datetime):
        t = dt.astimezone(tz=datetime.timezone.utc).timestamp()
        os.utime(self.path, times=(t, t))
        self.mtime = dt
        return self
    def to_rdd(self, headers=True, inferSchema=True):
        """
        Return the result set as a DataFrame

        Parameters:

            index(:obj:`str`, optional): Column to be used as the index of the DataFrame, defaults to :obj:`None`



        """
        try:
            import pyspark
        except ModuleNotFoundError as error:
            raise error


        from pyspark.sql import SparkSession

        t = Timer()
        t.start()
        sc = SparkSession.builder.getOrCreate()
        sc.conf.set("spark.sql.execution.arrow.enabled", "true")


        if pathlib.PurePosixPath(self.path).parts[1] == 'dbfs':
            dbfs_path = self.path.replace('/dbfs/', '')
        else:
            dbfs_path = self.path

        rdd = sc.read.format('csv').options(header=headers, inferSchema=inferSchema).load(dbfs_path)

        n_rows = rdd.count()
        n_cols = len(rdd.columns)
        t.stop(n_items=n_rows,
               n_bytes=self.bytes,
               item_name='rows'
               )
        self.log.info("Returned RDD({n_cols}x{n_rows}), "
                      "{bytes_str} in {duration}, "
                      "{bits_sec}, "
                      "{bytes_sec}, "
                      "{bytes_sec}".format(n_cols=n_rows,
                                           n_rows=n_cols,
                                           bytes_str=t.bytes_str,
                                           duration=t.duration,
                                           bits_sec=t.bits_per_second,
                                           bytes_sec=t.bytes_per_second_str,
                                           rows_sec=t.items_per_second_str
                                           ))


        return rdd

    def to_df(self, headers=True,
              columns: (bool, str, None, list, tuple) = None,
              parse_dates: (bool, str, None, list, tuple) =True,
              index_col: (str, None, list, tuple) = None):
        """
        Return the FTPFile as a Pandas DataFrame

        Parameters:

            index(:obj:`str`, optional): Column to be used as the index of the DataFrame, defaults to :obj:`None`

        Returns:
            :class:`pd.DataFrame`: Pandas DataFrame

        """

        # https://pandas.pydata.org/pandas-docs/version/0.24.2/reference/api/pandas.read_csv.html#pandas.read_csv
        try:
            import pandas as pd
        except ModuleNotFoundError as error:
            raise error

        if columns is not None and isinstance(columns, (list, tuple)) is True and len(columns) > 0:
            columns = columns
            usecols = columns
        else:
            columns = None
            usecols = None

        if headers is True:
            header = 'infer'
        if headers is False:
            header = 0
        else:
            header = 'infer'

        if parse_dates is True or (isinstance(parse_dates, (list, tuple)) is True and len(parse_dates) > 0):
            infer_datetime_format = True
        else:
            infer_datetime_format = False

        t = Timer()
        t.start()
        df = pd.read_csv(self.path, sep=',',
                         header=header,
                         names=columns,
                         usecols=usecols,
                         index_col=index_col,
                         parse_dates=parse_dates,
                         infer_datetime_format=infer_datetime_format,
                         memory_map=True,
                         low_memory=False,
                         compression='infer'
                         )
        df.fillna(pd.np.nan, inplace=True)
        n_rows = df.shape[0]
        n_cols = len(df.columns)
        n_bytes = df.memory_usage(index=True, deep=False).sum()
        t.stop(n_items=int(n_rows),
               n_bytes=int(n_bytes),
               item_name='rows'
               )
        self.log.info("Returned DataFrame(cols={n_cols}, rows={n_rows}, size={bytes_str}) "
                      "in: '{duration}', '{bits_sec}', '{bytes_sec}', '{rows_sec}'".format(n_cols=n_cols,
                                                                                           n_rows=n_rows,
                                                                                           bytes_str=t.bytes_str,
                                                                                           duration=t.duration,
                                                                                           bits_sec=t.megabits_per_second_str,
                                                                                           bytes_sec=t.bytes_per_second_str,
                                                                                           rows_sec=t.items_per_second_str
                                                                                           ))

        return df


@dataclasses.dataclass(init=True, repr=False, eq=True, order=True, unsafe_hash=False, frozen=False)
class CachedFile(LocalFile):
    path: str = dataclasses.field(init=True)
    directory: str = dataclasses.field(init=False)
    name: str = dataclasses.field(init=False)
    exists: bool = dataclasses.field(init=False)
    size: str = dataclasses.field(init=False)
    bytes: int = dataclasses.field(init=False)
    extension: str = dataclasses.field(init=False)
    ctime: datetime.datetime = dataclasses.field(init=False)
    mtime: datetime.datetime = dataclasses.field(init=False)
    _md5: str = dataclasses.field(init=False, default=None)

    def __repr__(self,
                 size=True,
                 mtime=False,
                 ctime=False,
                 exists=True
                 ):
        if self.exists is True:
            msgs = []
            if exists is True and self.exists is not None:
                msgs.append(f"exists={self.exists}")
            if size is True and self.size is not None:
                msgs.append(f"size='{self.size}'")
            if mtime is True and self.mtime is not None:
                msgs.append(f"mtime='{self.mtime}'")
            if ctime is True and self.ctime is not None:
                msgs.append(f"ctime='{self.ctime}'")

            if len(msgs) > 0:
                msgs_str = ", ".join(msgs)
                return f"LocalFile('{self.path}', {msgs_str})"
            else:
                return f"LocalFile('{self.path}')"
        else:
            return f"{self.__class__.__name__}('{self.path}', exists=False)"

    def __str__(self,
                size=True,
                mtime=False,
                ctime=False,
                exists=True
                ):
        return self.__repr__(
                size=True,
                mtime=False,
                ctime=False,
                exists=True
                )


class FTPMethods(object):
    log = get_logger()

    @classmethod
    def _get_obj(cls, ftp: ftplib.FTP_TLS, path: str, classname=None):
        parts = pathlib.PurePosixPath(path)
        for obj in ftp.mlsd(parts.parent):
            if obj[0] == parts.name and obj[1]['create'] != '19700131000000.000':
                if obj[1]['type'] == 'file':
                    return FTPFile.from_obj(ftp=ftp, obj=obj, parent_dir=str(parts.parent), classname=classname)
                elif obj[1]['type'] == 'dir':
                    return FTPFolder.from_obj(ftp=ftp, obj=obj, parent_dir=str(parts.parent), classname=classname)

                return obj
        return None

    @classmethod
    def _get_raw_object(cls, ftp: ftplib.FTP_TLS, path: str, type: str, classname: str = None):

        parts = pathlib.PurePosixPath(path)
        for obj in ftp.mlsd(parts.parent):
            if obj[0] == parts.name and obj[1]['create'] != '19700131000000.000' and obj[1]['type'] == type:
                return obj
        if str(type) == 'file':
            obj = (parts.name, {
                    'create': '19700131000000.000',
                    'modify': '19700131000000.000',
                    'type':   'file',
                    'size':   str(0)
                    }
                   )
            if classname is not None and isinstance(classname, str) and len(classname) > 0:
                obj[1]['classname'] = classname

            return obj
        if str(type) == 'dir':
            obj = (parts.name, {
                    'create': '19700131000000.000',
                    'modify': '19700131000000.000',
                    'type':   'dir',
                    'size':   str(0)
                    }
                   )
            if classname is not None and isinstance(classname, str) and len(classname) > 0:
                obj[1]['classname'] = classname
            return obj
        return None

    @classmethod
    def _get_file(cls,
                  ftp: ftplib.FTP_TLS,
                  path: str,
                  classname: str = None
                  ):


        return FTPFile.from_path(ftp=ftp, path=path, classname=classname)

    @classmethod
    def _get_folder(cls, ftp: ftplib.FTP_TLS, path: str, classname: str = None):

        return FTPFolder.from_path(ftp=ftp, path=path, classname=classname)

    @classmethod
    def _path_exists(cls, ftp: ftplib.FTP_TLS, path: str):
        parts = pathlib.PurePosixPath(path)
        if path == "/":
            return True
        for f in ftp.mlsd(parts.parent):
            if f[0] == parts.name and f[1]['create'] != '19700131000000.000':
                return True
        return False

    @classmethod
    def _dir_exists(cls, ftp: ftplib.FTP_TLS, path: str):
        parts = pathlib.PurePosixPath(path)
        for f in ftp.mlsd(parts.parent):
            if f[0] == parts.name and f[1]['create'] != '19700131000000.000' and f[1]['type'] == 'dir':
                return True
        return False

    @classmethod
    def _file_exists(cls, ftp: ftplib.FTP_TLS, path: str):
        parts = pathlib.PurePosixPath(path)
        for f in ftp.mlsd(parts.parent):
            if f[0] == parts.name and f[1]['create'] != '19700131000000.000' and f[1]['type'] == 'file':
                return True
        return False

    @classmethod
    def __list_objs(cls, ftp: ftplib.FTP_TLS, path: str = '.'):
        items = []
        if path == '.':
            path = '/'
        if len(path) > 1 and path[-1] != '/':
            path = path + "/"
        parent_dir = path
        for obj in ftp.mlsd(path):
            if obj[1] not in ('.', '..'):
                items.append(obj)
        items = sorted(items, key=lambda x: x[1]['modify'], reverse=True)
        return items

    @classmethod
    def _list_objs(cls,
                   ftp: ftplib.FTP_TLS,
                   path: str = '.',
                   file_classname: str = None,
                   folder_classname: str = None
                   ):

        items = []
        if path == '.':
            path = '/'
        if len(path) > 1 and path[-1] != '/':
            path = path + "/"
        parent_dir = path
        for obj in ftp.mlsd(path):
            if obj[0] not in ('.', '..'):
                if obj[1]['type'] == 'file':
                    file = FTPFile.from_obj(ftp=ftp, obj=obj, parent_dir=parent_dir, classname=file_classname)
                    items.append(file)
                if obj[1]['type'] == 'dir':
                    folder = FTPFolder.from_obj(ftp=ftp, obj=obj, parent_dir=parent_dir, classname=folder_classname)
                    items.append(folder)
        items = sorted(items, key=lambda x: x.mtime, reverse=True)
        return items

    @classmethod
    def _list_files(cls, ftp: ftplib.FTP_TLS, path: str = '.', classname: str = None):
        items = []
        for item in cls._list_objs(ftp=ftp, path=path, file_classname=classname, folder_classname=None):
            if item.type == 'file':
                items.append(item)
        return items

    @classmethod
    def _list_dirs(cls, ftp: ftplib.FTP_TLS, path: str = '.', classname: str = None):
        items = []
        for item in cls._list_objs(ftp=ftp, path=path, file_classname=None, folder_classname=classname):
            if item.type == 'dir':
                items.append(item)
        return items

    @classmethod
    def _list_filenames(cls, ftp: ftplib.FTP_TLS, path: str = '.', classname: str = None):
        folders = []
        for file in cls._list_files(ftp, path=path, classname=classname):
            folders.append(file.name)
        return folders

    @classmethod
    def _list_dirnames(cls, ftp: ftplib.FTP_TLS, path: str = '.', classname: str = None):
        folders = []
        for folder in cls._list_dirs(ftp, path=path, classname=classname):
            folders.append(folder.name)
        return folders

    @classmethod
    def _makedirs(cls, ftp: ftplib.FTP_TLS, path: str, classname: str = None):
        parts = pathlib.PurePosixPath(path)
        if cls._path_exists(ftp, path) is False:
            iter_dir = "/"
            ftp.cwd(iter_dir)
            for subdir in parts.parent.parts[1:]:
                folders = [obj for obj in ftp.mlsd(iter_dir) if obj[1]['type'] == 'dir']
                folder_names = [n[0] for n in folders]
                if subdir not in folder_names:
                    # cls.log.info(f"Directory does not exist yet: {subdir}")
                    ftp.mkd(subdir)
                    # cls.log.info(f"Created directory: {subdir}")
                ftp.cwd(subdir)
                iter_dir = iter_dir + "/" + subdir
            # cls.log.info(f"Created directory: {iter_dir}")
        folder = cls._get_obj(ftp, path, classname=classname)
        return folder

    @classmethod
    def _get_temp_path(cls,
                       ftp: ftplib.FTP_TLS,
                       remote_path: str,
                       remote_file=None,
                       classname: str = None
                       ):

        if remote_file is not None and hasattr(remote_file, 'ftp'):
            remote_file = remote_file
        else:
            remote_file = FTPFile.from_path(ftp=ftp, path=remote_path, classname=classname)

        temp_dir = gettempdir()
        temp_dir_lib = os.path.join(temp_dir, 'ftplib')
        temp_dir_host = os.path.join(temp_dir_lib, ftp.host)

        parts = pathlib.PurePosixPath(remote_path)
        name = parts.stem + '__' + remote_file.md5 + parts.suffix
        path = os.path.join(temp_dir_host, name)

        return path

    @classmethod
    def _download_temp_file(cls,
                            ftp: ftplib.FTP_TLS,
                            remote_path: str,
                            remote_file=None,
                            set_ctime=False,
                            set_mtime=True,
                            classname: str = None
                            ):

        if remote_file is not None and hasattr(remote_file, 'ftp'):
            remote_file = remote_file
        else:
            remote_file = FTPFile.from_path(ftp=ftp, path=remote_path, classname=classname)

        remote_file.assert_exists()

        temp_dir = gettempdir()
        temp_dir_lib = os.path.join(temp_dir, 'ftplib')
        temp_dir_host = os.path.join(temp_dir_lib, ftp.host)

        parts = pathlib.PurePosixPath(remote_file.path)
        name = parts.stem + '__' + remote_file.md5 + parts.suffix
        path = os.path.join(temp_dir_host, name)

        if os.path.exists(path) is True:
            temp_file = CachedFile(path=path)
            print(f"{temp_file.__repr__(size=True)}: is already cached")
            return CachedFile(path=path)
        new_file = cls._download_file(ftp=ftp,
                                       local_path=path,
                                       remote_path=remote_path,
                                       remote_file=remote_file,
                                       set_ctime=set_ctime,
                                       set_mtime=set_mtime,
                                       classname=classname
                                       )

        temp_file = CachedFile(path=path)
        return temp_file

    @classmethod
    def _download_file(cls,
                       ftp: ftplib.FTP_TLS,
                       remote_path: str,
                       local_path: str,
                       remote_file=None,
                       set_ctime=False,
                       set_mtime=True,
                       classname: str = None
                       ):

        if remote_file is not None and hasattr(remote_file, 'ftp'):
            remote_file = remote_file
        else:
            remote_file = FTPFile.from_path(ftp=ftp, path=remote_path, classname=classname)

        local_file = LocalFile(local_path)

        remote_file.assert_exists()

        if local_file.exists is True:
            remote_md5 = remote_file.md5
            local_md5 = local_file.md5
            copy_local_file = LocalFile(path=local_path)

            if local_file.mtime > remote_file.mtime:
                cls.log.warning(f"{local_file.__repr__(mtime=True, size=True)} "
                                f"is newer than: {remote_file.__repr__(mtime=True, size=True)} "
                                f"by: {str(local_file.mtime - remote_file.mtime)}")
            if local_file.size > remote_file.size:
                cls.log.warning(f"{local_file.__repr__(size=True, exists=False)} "
                                f"is larger than: {remote_file.__repr__(size=True, exists=False)} "
                                f"by: {humansize(local_file.bytes - remote_file.bytes)}")

            if local_md5 == remote_md5:
                cls.log.info(f"{local_file.__repr__(size=False, exists=False)} "
                             f"and {remote_file.__repr__(exists=False, size=False)} "
                             f"have the same MD5: '{local_md5}'")
                cls.log.info(
                        f"No files were downloaded. "
                        f"{local_file.__repr__(exists=False, size=False)} "
                        f"and {remote_file.__repr__(size=False, exists=False)} "
                        f"are the same file.")
                return local_file
            if local_md5 != remote_md5:
                remote_file = remote_file.remove()
                cls.log.info(f"Removed existing: {copy_local_file.__repr__(size=True, exists=False)}")

        local_file.makedirs()
        with Timer() as t:
            with open(local_file.path, 'wb') as outfile:
                # ftp.cwd("/")
                # ftp.cwd(remote_file.directory)
                ftp.retrbinary(f'RETR {remote_file.path}', outfile.write)

        new_local_file = LocalFile(local_path)

        if new_local_file.exists is False:
            t.stop(n_bytes=remote_file.bytes)
            cls.log.error(f"Failed to download: {remote_file.__repr__(exists=True, size=True)} "
                          f"to {new_local_file.__repr__(exists=False, size=False)}")
            new_local_file.assert_exists()
        if new_local_file.exists is True:
            if set_mtime is True:
                new_local_file = new_local_file.set_mtime(dt=remote_file.mtime)
            if set_ctime is True:
                new_local_file = new_local_file.set_ctime(dt=remote_file.ctime)
            t.stop(n_bytes=new_local_file.bytes)
            cls.log.info(
                    f"Downloaded {remote_file.__repr__(exists=False, size=True)} "
                    f"to {new_local_file.__repr__(size=False, exists=False)} "
                    f"in: '{t.duration_str}', "
                    f"('{t.bytes_str}', "
                    f"'{t.megabits_per_second_str}', "
                    f"'{t.megabytes_per_second_str}'")
            return new_local_file

    @classmethod
    def _upload_file(cls,
                     ftp: ftplib.FTP_TLS,
                     remote_path: str,
                     local_path: str,
                     remote_file=None,
                     set_ctime: bool = False,
                     set_mtime: bool = True,
                     classname: str = None
                     ):

        if remote_file is not None and hasattr(remote_file, 'ftp'):
            remote_file = remote_file
        else:
            remote_file = FTPFile.from_path(ftp=ftp, path=remote_path, classname=classname)

        local_file = LocalFile(path=local_path)
        local_file.assert_exists()

        cls.log.info(f"Uploading: {local_file.__repr__(size=True, exists=False)} "
                     f"to {remote_file.__repr__(exists=True, size=False)}")

        if remote_file.exists is True:
            remote_md5 = remote_file.md5
            local_md5 = local_file.md5
            copy_remote_file = remote_file.copy_obj()
            if remote_file.mtime > local_file.mtime:
                cls.log.warning(f"{remote_file.__repr__(mtime=True, size=True)} "
                                f"is newer than: {local_file.__repr__(mtime=True, size=True)} "
                                f"by: {str(remote_file.mtime - local_file.mtime)}")
            if remote_file.size > local_file.size:
                cls.log.warning(f"{remote_file.__repr__(size=True)} "
                                f"is larger than: {local_file.__repr__(size=True)} "
                                f"by: {humansize(remote_file.bytes - local_file.bytes)}")

            if local_md5 == remote_md5:
                cls.log.info(f"{remote_file.__repr__(size=False, exists=False)} "
                             f"and {local_file.__repr__(exists=False,size=False)} "
                             f"have the same MD5: '{local_md5}'")
                cls.log.info(
                    f"No files were uploaded. {remote_file.__repr__(exists=False, size=False)} "
                    f"and {local_file.__repr__( size=False, exists=False)} "
                    f"are the same file.")
                return remote_file
            if local_md5 != remote_md5:
                remote_file = remote_file.remove()
                cls.log.info(f"Removed existing: {copy_remote_file.__repr__(size=True, exists=False)}")

        remote_file.makedirs()
        with Timer() as t:
            with open(local_file.path, 'rb') as infile:
                # ftp.cwd("/")
                # ftp.cwd(remote_file.directory)
                ftp.storbinary(f'STOR {remote_file.path}', infile)
        new_remote_file = FTPFile.from_path(ftp=ftp, path=remote_path, classname=classname)

        if new_remote_file.exists is False:
            t.stop()
            cls.log.error(f"Failed to upload: {local_file.__repr__(size=True, exists=True)} "
                          f"to {new_remote_file.__repr__(exists=False, size=False)}")
            new_remote_file.assert_exists()
        if new_remote_file.exists is True:
            if set_mtime is True:
                new_remote_file = new_remote_file.set_mtime(dt=local_file.mtime)
            if set_ctime is True:
                new_remote_file = new_remote_file.set_ctime(dt=local_file.ctime)
            remote_file.update_metadata(new_remote_file)
            t.stop(n_bytes=new_remote_file.bytes)
            cls.log.info(f"Uploaded: {local_file.__repr__(size=True, exists=False)} "
                         f"to {new_remote_file.__repr__(exists=False, size=False)} "
                         f"in: '{t.duration_str}', "
                         f"'{t.megabits_per_second_str}', "
                         f"'{t.megabytes_per_second_str}'")
            return new_remote_file

    @classmethod
    def _read_file(cls,
                   ftp: ftplib.FTP_TLS,
                   path: str,
                   file=None,
                   classname: str = None
                   ):

        if file is not None and hasattr(file, 'ftp'):
            file = file
        else:
            file = FTPFile.from_path(ftp=ftp, path=path, classname=classname)


        bytesio = BytesIO()
        file.assert_exists()

        cls.log.info(f"Reading: {file.__repr__(exists=False, size=True)}")

        # ftp.cwd("/")
        # ftp.cwd(file.directory)

        with Timer() as t:
            ftp.retrbinary(f'RETR {file.path}', bytesio.write)
            string = bytesio.getvalue().decode('utf-8')
            t.stop(n_bytes=getsizeof(string))

        cls.log.info(f"Read {len(string.splitlines())} lines, "
                     f"{len(string)} chars "
                     f"from: {file.__repr__(exists=False, size=True)} "
                     f"in: '{t.duration_str}', "
                     f"'{t.megabits_per_second_str}',"
                     f"'{t.megabytes_per_second_str}'"
                     )

    @classmethod
    def _write_file(cls,
                    ftp: ftplib.FTP_TLS,
                    path: str,
                    string: str,
                    ctime: datetime.datetime = None,
                    mtime: datetime.datetime = None,
                    file=None,
                    classname: str = None
                    ):

        if file is not None and hasattr(file, 'ftp'):
            file = file
        else:
            file = FTPFile.from_path(ftp=ftp, path=path, classname=classname)

        cls.log.info(f"Writing: {file.__repr__(size=False, exists=False)} ")
        if file.exists is True:
            copy_remote_file = file.copy_obj()
            file = file.remove()
            cls.log.info(f"Removed existing: {copy_remote_file.__repr__(size=False, exists=False)}")

        file.makedirs()
        ftp.cwd("/")
        ftp.cwd(file.directory)
        with Timer() as t:
            ftp.storbinary(f'STOR {file.name}', BytesIO(string.encode('utf-8')))
            t.stop(n_bytes=getsizeof(string))

        new_file = FTPFile.from_path(ftp=ftp, path=path, classname=classname)

        if mtime is not None and isinstance(mtime, datetime.datetime):
            new_file = new_file.set_mtime(dt=mtime)
        if ctime is not None and isinstance(mtime, datetime.datetime):
            new_file = new_file.set_mtime(dt=ctime)
        file.update_metadata(new_file)
        if new_file.exists is False:
            cls.log.error(f"Failed to write "
                          f"{len(string.splitlines())} lines, "
                          f"{len(string)} chars "
                          f"to: {file.__repr__(exists=False, size=False)}")
            new_file.assert_exists()
        if new_file.exists is True:
            cls.log.info(f"Wrote {len(string.splitlines())} lines, "
                         f"{len(string)} chars "
                         f"to: {new_file.__repr__(size=True, exists=False)} "
                         f"in: '{t.duration_str}', "
                         f"'{t.megabits_per_second_str}', "
                         f"'{t.megabytes_per_second_str}'"
                         )
            return new_file

    @classmethod
    def _delete_file(cls,
                     ftp: ftplib.FTP_TLS,
                     path: str,
                     file=None,
                     classname: str = None
                     ):

        if file is not None and hasattr(file, 'ftp'):
            file = file
        else:
            file = FTPFile.from_path(ftp=ftp, path=path, classname=classname)

        if file.exists is True:
            old_file = file.copy_obj()
            with Timer() as t:
                ftp.cwd("/")
                ftp.cwd(file.directory)
                ftp.delete(file.name)
                t.stop()
            file = file.reset()
            cls.log.info(f"Removed: {old_file.__repr__(exists=False, size=True)} in: '{t.duration_str}'")

            return file
        else:
            return file

    @classmethod
    def _rename_file(cls,
                     ftp: ftplib.FTP_TLS,
                     path: str,
                     name: str,
                     file=None,
                     classname: str = None
                     ):

        if file is not None and hasattr(file, 'ftp'):
            file = file
        else:
            file = FTPFile.from_path(ftp=ftp, path=path, classname=classname)

        if file.exists is True:
            new_file = file.copy_obj()
            with Timer() as t:
                ftp.cwd("/")
                ftp.cwd(file.directory)
                ftp.rename(file.name, name)
                t.stop()
            new_file.name = name
            file = file.reset()
            cls.log.info(f"Renamed: {file.__repr__(exists=True, size=False)} "
                         f"to: {new_file.__repr__(exists=True, size=False)} "
                         f"in: '{t.duration_str}'")

            return new_file

        else:
            return file

    @classmethod
    def _delete_dir(cls,
                    ftp: ftplib.FTP_TLS,
                    path: str,
                    folder=None,
                    classname: str = None
                    ):

        if folder is not None and hasattr(folder, 'ftp'):
            folder = folder
        else:
            folder = FTPFolder.from_path(ftp=ftp, path=path, classname=classname)

        folder.assert_exists()
        # TODO: Fix this
        new_folder = folder.copy_obj()
        with Timer() as t:
            ftp.cwd("/")
            ftp.cwd(folder.directory)
            ftp.rmd(folder.name)
            t.stop()
        new_folder.reset()
        cls.log.info(f"Removed: {folder.__repr__(size=True)} in: '{t.duration_str}'")
        return new_folder

    @classmethod
    def _connect(cls,
                 host: str,
                 port: int,
                 username: str,
                 password: str,
                 timeout: int = 120
                 ):
        try:
            with Timer() as t:
                context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
                context.verify_mode = ssl.CERT_REQUIRED
                # context.check_hostname = True
                context.load_default_certs()

                ftp = ftplib.FTP_TLS(timeout=10, context=context)
                ftp.connect(host=host, port=port, timeout=10)
                ftp.auth()
                ftp.prot_p()
                ftp.set_pasv(True)
                ftp.login(user=username, passwd=password, secure=True)
                ftp.timeout = timeout
                t.stop()
            cls.log.info(f"{username} Connected to '{host}:{port}' in '{t.duration_str}'")

            return ftp
        except ftplib.all_errors as e:
            raise e

    @classmethod
    def _is_connected(cls,
                      ftp: ftplib.FTP_TLS
                      ):
        if ftp is not None and isinstance(ftp, ftplib.FTP_TLS):
            try:
                cwd = ftp.pwd()
                return True
            except Exception as e:
                return False
        else:
            return False

    @classmethod
    def _close(cls,
               ftp: ftplib.FTP_TLS
               ):
        if cls._is_connected(ftp) is True:
            ftp.quit()

    @classmethod
    def _available_commands(cls,
                            ftp: ftplib.FTP_TLS
                            ):
        return [x.strip() for x in ftp.sendcmd('Feat').splitlines()[1:-1]]

    def _set_modifed_dt(self,
                        ftp: ftplib.FTP_TLS,
                        path: str,
                        mtime: datetime.datetime,
                        file=None,
                        classname: str = None
                        ):

        if file is not None and hasattr(file, 'ftp'):
            file = file
        else:
            file = FTPFile.from_path(ftp=ftp, path=path, classname=classname)

        if file.exists is True:
            # ftp.cwd('/')
            # ftp.cwd(file.directory)
            dt_str = datetime_to_ftptime(mtime)
            ftp.sendcmd(f"MFMT {dt_str} {file.path}")
            file.mtime = mtime
        return file

    def _set_ctime(self,
                   ftp: ftplib.FTP_TLS,
                   path: str,
                   ctime: datetime.datetime,
                   file=None,
                   classname: str = None
                   ):

        if file is not None and hasattr(file, 'ftp'):
            file = file
        else:
            file = FTPFile.from_path(ftp=ftp, path=path, classname=classname)
        if file.exists is True:
            # ftp.cwd('/')
            # ftp.cwd(file.directory)
            dt_str = datetime_to_ftptime(ctime)
            ftp.sendcmd(f"MFCT {dt_str} {file.path}")
            file.mtime = ctime
        return file

    def _get_md5(self,
                 ftp: ftplib.FTP_TLS,
                 path: str,
                 file=None,
                 classname: str = None
                 ):

        if file is not None and hasattr(file, 'ftp'):
            file = file
        else:
            file = FTPFile.from_path(ftp=ftp, path=path, classname=classname)
        if file.exists is True:
            md5_str = str(ftp.sendcmd(f"MD5 {file.path}")).split(" ")[-1]
            return md5_str.upper()
        else:
            return None

    def _get_multiple_md5s(self,
                           ftp: ftplib.FTP_TLS,
                           paths: (list, tuple),
                           files: (list, tuple),
                           is_box: bool = False
                           ):
        if is_box is True:
            FileClassType = BoxFile
        else:
            FileClassType = BoxFile

        if files is not None and len(files) == len(paths):
            files = files
        else:
            files = []
            for path in paths:
                file = FileClassType.from_path(ftp=ftp, path=path)
                files.append(file)

        new_files = []
        for file in files:
            if file.exists is True:
                new_files.append(file)
        paths_str = ",".join([file.path for file in new_files])
        nnew_files = []
        if len(new_files) > 0:
            resp = str(ftp.sendcmd(f"MMD5 {paths_str}"))
            RE_MMD5 = re.compile(r".*([A-Za-z0-9]{32}).*")
            for idx, part in enumerate(resp.split(", ")):
                match = RE_MMD5.findall(part)[0]
                new_file = new_files[idx]
                new_file._md5 = match
                nnew_files.append(new_file)
            return nnew_files
        else:
            return nnew_files


class FTPClientMethods(FTPMethods):
    host: str = NotImplemented
    port: int = NotImplemented
    username: str = NotImplemented
    password: str = NotImplemented
    timeout: int = NotImplemented
    ftp: ftplib.FTP_TLS = NotImplemented


    file_classname: str = None
    folder_classname: str = None

    def __init__(self, host: str, port: int, username: str, password: str, file_classname: str, folder_classname: str ):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.host}:{self.port}')"

    @property
    def closed(self):
        if self._is_connected(ftp=self.ftp) is True:
            return False
        else:
            return True

    def close(self):
        if self.closed is False:
            self._close(ftp=self.ftp)
            self.log.info(f"{self.__repr__()}: Closed Connection")

    def connect(self,
                ftp: ftplib.FTP_TLS = None
                ):
        if ftp is not None and isinstance(ftp, ftplib.FTP_TLS):
            if self._is_connected(ftp) is True:
                self.ftp = ftp
        else:
            self.ftp = self._connect(host=self.host,
                                     username=self.username,
                                     password=self.password,
                                     port=self.port,
                                     timeout=self.timeout
                                     )
            return self.ftp

    def reset(self):
        if self.closed is False:
            self.close()
        self.connect(ftp=None)

    def new_connection(self):
        return FTP(username=self.username,
                   password=self.password,
                   host=self.host,
                   port=self.port,
                   timeout=self.timeout
                   )

    def file(self, path: str):
        return FTPFile.from_path(ftp=self.ftp, path=path, classname=self.file_classname )

    def folder(self, path: str):
        return FTPFolder.from_path(ftp=self.ftp, path=path, classname=self.folder_classname)

    def files(self, path: str = '.'):
        if path is None:
            path = '/'
        return self._list_files(ftp=self.ftp, path=path, classname=self.file_classname)


    def folders(self, path: str = '.'):
        if path is None:
            path = '/'
        return self._list_dirs(ftp=self.ftp, path=path,  classname=self.folder_classname)

    def list(self, path: str = '.'):
        if path is None:
            path = '/'
        return self._list_objs(ftp=self.ftp, path=path, file_classname=self.file_classname, folder_classname=self.folder_classname)

    def upload_file(self,
                    remote_path: str,
                    local_path: str,
                    remote_file=None,
                    set_ctime: bool = False,
                    set_mtime: bool = True,
                    ):
        return self._upload_file(ftp=self.ftp,
                                 remote_path=remote_path,
                                 local_path=local_path,
                                 remote_file=remote_file,
                                 set_ctime=set_ctime,
                                 set_mtime=set_mtime,
                                 classname=self.file_classname
                                 )

    def download_file(self,
                      remote_path: str,
                      local_path: str,
                      remote_file=None,
                      set_ctime=False,
                      set_mtime=True
                      ):
        return self._download_file(ftp=self.ftp,
                                   remote_path=remote_path,
                                   local_path=local_path,
                                   remote_file=remote_file,
                                   set_ctime=set_ctime,
                                   set_mtime=set_mtime,
                                   classname=self.file_classname
                                   )

    def download_temp_file(self,
                           remote_path: str,
                           remote_file=None,
                           set_ctime=False,
                           set_mtime=True,
                           is_box: bool = False
                           ):
        return self._download_temp_file(ftp=self.ftp,
                                        remote_path=remote_path,
                                        remote_file=remote_file,
                                        set_ctime=set_ctime,
                                        set_mtime=set_mtime,
                                        classname=self.file_classname
                                        )


@dataclasses.dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=False, frozen=False)
class FTP(FTPClientMethods):
    # https://docs.python.org/3/library/ftplib.html#ftplib.FTP
    host: str = dataclasses.field(init=True)
    port: int = dataclasses.field(init=True)
    username: str = dataclasses.field(init=True)
    password: str = dataclasses.field(init=True)
    timeout: int = dataclasses.field(init=True, default=120)
    ftp: ftplib.FTP_TLS = dataclasses.field(init=False)

    file_classname: str = None
    folder_classname: str = None

    def __post_init__(self):
        self.connect(ftp=None)


class FTPFileMethods(FTPMethods):
    ftp: ftplib.FTP_TLS = NotImplemented
    path: str = NotImplemented

    directory: str = NotImplemented
    name: str = NotImplemented
    extension: str = NotImplemented

    bytes: int = NotImplemented
    size: str = NotImplemented

    # md5: str = dataclasses.field(init=False)
    ctime: datetime.datetime = NotImplemented
    mtime: datetime.datetime = NotImplemented
    exists: bool = NotImplemented
    type: str = NotImplemented
    _md5: str = NotImplemented

    obj: dataclasses.InitVar[tuple] = NotImplemented
    classname: dataclasses.InitVar[str] = NotImplemented
    _classname: str = NotImplemented

    def __init__(self, ftp: ftplib.FTP_TLS, path: str, obj: tuple, classname: str):
        raise NotImplementedError

    def __repr__(self,
                 size: bool = True,
                 mtime: bool = False,
                 ctime: bool = False,
                 exists: bool = True
                 ):
        if self.exists is True:
            msgs = []
            if exists is True and self.exists is not None:
                msgs.append(f"exists={self.exists}")
            if size is True and self.size is not None:
                msgs.append(f"size='{self.size}'")
            if mtime is True and self.mtime is not None:
                msgs.append(f"mtime='{self.mtime}'")
            if ctime is True and self.ctime is not None:
                msgs.append(f"ctime='{self.ctime}'")

            if len(msgs) > 0:
                msgs_str = ", ".join(msgs)
                return f"{self.__class__.__name__}('{self.path}', {msgs_str})"
            else:
                return f"{self.__class__.__name__}('{self.path}')"
        else:
            return f"{self.__class__.__name__}('{self.path}', exists=False)"
            # return f"FTPFile('{self.path}', exists={self.exists})"

    @property
    def classname(self):
        if self._classname is not None and isinstance(self._classname, str) and len(self._classname) > 0:
            return self._classname
        else:
            return self.__class__.__name__

    def get_classname(self, classname: str = None):
        if classname is not None and isinstance(classname, str) and len(classname) > 0:
            return classname
        return None

    def set_classname(self, classname: str = None):
        self._classname = self.get_classname(classname)

    def rename_class(self, classname: str = None):
        og_classname = self.__class__.__name__
        new_classname = self.get_classname(classname)
        if new_classname is not None:
            self.__doc__ = (self.__class__.__name__ +
                            str(inspect.signature(self.__class__)).replace(' -> None', ''))
            self.__class__.__name__ = classname
            self.__class__.__qualname__ = classname

    def __str__(self,
                size: bool = True,
                mtime: bool = False,
                ctime: bool = False,
                exists: bool = True
                ):
        return self.__repr__(size=size, mtime=mtime, ctime=ctime, exists=exists)

    def __post_init__(self, obj: tuple, classname: str):
        self._classname = classname
        self.initialize(obj=obj, classname=classname)


    def initialize(self, obj: tuple, classname: str = None):

        parts = pathlib.PurePosixPath(self.path)
        self.directory = str(parts.parent)
        self.name = str(parts.name)
        self.extension = str(parts.suffix)

        if obj is None:
            obj = self._get_raw_object(ftp=self.ftp, path=self.path, type='file')

        self.exists = False
        if obj is not None:
            assert obj[0] == self.name
            assert obj[1]['type'] == 'file'
            assert 'modify' in obj[1]
            assert 'create' in obj[1]
            if obj[1]['modify'] != '19700131000000.000' and obj[1]['create'] != '19700131000000.000':
                self.exists = True
            self.bytes = int(obj[1]['size'])
            self.size = humansize(self.bytes)
            self.ctime = parse_time(obj[1]['create'])
            self.mtime = parse_time(obj[1]['modify'])

            # TODO: Add call to get MD5

            self.rename_class(classname=classname)


    @classmethod
    def from_obj(cls,
                 ftp: ftplib.FTP_TLS,
                 obj: tuple,
                 parent_dir: str,
                 classname: str = None
                 ):
        """

        from an ftplib.File object
            ex. ('Sapphire End User Documentation',
                    {   'size': '7327748',
                        'modify': '20191018204619.000',
                        'create': '20180412221908.000',
                        'type': 'dir'
                    }
                )
        """

        assert obj[1]['type'] == 'file'
        if parent_dir[-1] != "/":
            parent_dir = parent_dir + "/"
        path = parent_dir + obj[0]

        return cls(ftp=ftp,
                   path=path,
                   obj=obj,
                   classname=classname
                   )


    def to_obj(self):

        obj = (self.name, {
                'create': datetime_to_ftptime(self.ctime),
                'modify': datetime_to_ftptime(self.mtime),
                'type':   'file',
                'size':   str(self.bytes)
                }
                )
        if self._md5 is not None:
            obj[1]['md5'] = self._md5

        obj[1]['classname'] = self.classname
        return obj



    @classmethod
    def from_path(cls,
                  ftp: ftplib.FTP_TLS,
                  path: str,
                  classname: str = None
                  ):
        obj = cls._get_raw_object(ftp=ftp, path=path, type='file', classname=classname)
        return cls(ftp=ftp, path=path, obj=obj, classname=classname)

    def refresh(self, obj: tuple, classname: str = None):
        self.clear()
        if classname is not None and isinstance(classname, str) and len(classname) > 0:
            obj[1]['classname'] = classname
        else:
            if self._classname is not None and isinstance(self._classname, str) and len(self._classname) > 0:
                obj[1]['classname'] = self._classname
        self.__post_init__(obj=obj, classname=classname)
        return self

    def clear(self):
        self.exists = False
        self.size = humansize(0)
        self.bytes = 0
        self.ctime = parse_time('19700131000000.000')
        self.mtime = parse_time('19700131000000.000')
        self._md5 = None
        return self

    def update_metadata(self, new_file):

        self.exists = new_file.exists
        self.mtime = new_file.mtime
        self.ctime = new_file.ctime
        self._md5 = new_file.md5
        self.path = new_file.path
        self.name = new_file.name
        self.directory = new_file.directory
        self.path = new_file.path
        self.extension = new_file.extension
        self.size = new_file.size

    def reset(self):
        self.clear()
        return self

    def assert_exists(self):
        if self.exists is False:
            raise FTPFileNotFoundError(path=self.path, classname=self.classname)


    def assert_not_exists(self):
        if self.exists is True:

            raise FTPFileExistsError(self.path, classname=self.classname)

    @property
    def md5(self):
        if self._md5 is not None and isinstance(self._md5, str):
            return self._md5
        else:
            self._md5 = self._get_md5(ftp=self.ftp, path=self.path, file=self,
                                      classname=self.classname)
        return self._md5

    def remove(self):
        if self.exists is True:
            file = self._delete_file(path=self.path, ftp=self.ftp, file=self,
                                     classname=self.classname)

            return file

        else:
            return self

    def makedirs(self):
        self._makedirs(ftp=self.ftp, path=self.directory, classname=self.classname)

    def rename(self, name):
        if self.exists is True:
            return self._rename_file(ftp=self.ftp, path=self.path, name=name, file=self,
                                     classname=self.classname)

    def read(self):
        return self._read_file(ftp=self.ftp, path=self.path, file=self,
                               classname=self.classname)

    def readlines(self):
        string = self.read()
        return string.splitlines()

    def write(self, string: str):
        return self._write_file(ftp=self.ftp, path=self.path, string=string, file=self,
                                classname=self.classname)

    def writelines(self, lines):
        if isinstance(lines, list) or isinstance(lines, tuple):
            string = "\n".join(lines)
        else:
            string = lines
        return self.write(string)

    def upload(self,
               path: str,
               set_mtime: bool = True,
               set_ctime: bool = False
               ):
        return self._upload_file(ftp=self.ftp,
                                 remote_path=self.path,
                                 local_path=path,
                                 remote_file=self,
                                 set_ctime=set_ctime,
                                 set_mtime=set_mtime,
                                 classname=self.classname
                                 )

    def download(self,
                 path: str,
                 set_mtime: bool = True,
                 set_ctime: bool = False
                 ):
        return self._download_file(ftp=self.ftp,
                                   local_path=path,
                                   remote_file=self,
                                   remote_path=self.path,
                                   set_ctime=set_ctime,
                                   set_mtime=set_mtime,
                                   classname=self.classname
                                   )

    def download_temp_file(self, set_mtime: bool = True,
                           set_ctime: bool = False):
        return self._download_temp_file(ftp=self.ftp,
                                        remote_file=self,
                                        remote_path=self.path,
                                        set_ctime=set_ctime,
                                        set_mtime=set_mtime,
                                        classname=self.classname
                                        )

    def set_mtime(self,
                  dt: datetime.datetime
                  ):
        file = self._set_modifed_dt(ftp=self.ftp,
                                    path=self.path,
                                    mtime=dt,
                                    file=self,
                                    classname=self.classname
                                    )
        self.mtime = file.mtime
        return file

    def set_ctime(self, dt: datetime.datetime):
        file = self._set_ctime(ftp=self.ftp,
                               path=self.path,
                               ctime=dt,
                               file=self,
                               classname=self.classname
                               )
        self.ctime = file.ctime
        return file

    def copy_obj(self):

        return FTPFile(ftp=self.ftp, path=self.path, obj=self.to_obj(), classname=self.classname)

    def to_rdd(self, headers=True, inferSchema=True):
        """
        Return the result set as a DataFrame

        Parameters:

            index(:obj:`str`, optional): Column to be used as the index of the DataFrame, defaults to :obj:`None`

        """
        try:
            import pyspark
        except ModuleNotFoundError as error:
            raise error

        temp_file = self.download_temp_file(set_mtime=True,
                                            set_ctime=False
                                            )

        from pyspark.sql import SparkSession

        t = Timer()
        t.start()
        sc = SparkSession.builder.getOrCreate()
        sc.conf.set("spark.sql.execution.arrow.enabled", "true")
        parts = pathlib.PurePosixPath(temp_file.path).parts[1]

        if pathlib.PurePosixPath(temp_file.path).parts[1] == 'dbfs':
            dbfs_path = temp_file.path.replace('/dbfs/', '')
        else:
            dbfs_path = temp_file.path

        rdd = sc.read.format('csv').options(header=headers, inferSchema=inferSchema).load(dbfs_path)

        n_rows = rdd.count()
        n_cols = len(rdd.columns)
        t.stop(n_items=n_rows,
               n_bytes=temp_file.bytes,
               item_name='rows'
               )
        self.log.info("Returned RDD({n_cols}x{n_rows}), "
                      "{bytes_str} in {duration}, "
                      "{bits_sec}, "
                      "{bytes_sec}, "
                      "{bytes_sec}".format(n_cols=n_rows,
                                           n_rows=n_cols,
                                           bytes_str=t.bytes_str,
                                           duration=t.duration,
                                           bits_sec=t.bits_per_second,
                                           bytes_sec=t.bytes_per_second_str,
                                           rows_sec=t.items_per_second_str
                                           ))
        return rdd

    def to_df(self, headers=True,
              columns: (bool, str, None, list, tuple) = None,
              parse_dates: (bool, str, None, list, tuple) =True,
              index_col: (str, None, list, tuple) = None):
        """
        Return the FTPFile as a Pandas DataFrame

        Parameters:

            index(:obj:`str`, optional): Column to be used as the index of the DataFrame, defaults to :obj:`None`

        Returns:
            :class:`pd.DataFrame`: Pandas DataFrame

        """

        # https://pandas.pydata.org/pandas-docs/version/0.24.2/reference/api/pandas.read_csv.html#pandas.read_csv
        try:
            import pandas as pd
        except ModuleNotFoundError as error:
            raise error

        temp_file = self.download_temp_file(set_mtime=True,
                                            set_ctime=False
                                            )

        if columns is not None and isinstance(columns, (list, tuple)) is True and len(columns) > 0:
            columns = columns
            usecols = columns
        else:
            columns = None
            usecols = None

        if headers is True:
            header = 'infer'
        if headers is False:
            header = 0
        else:
            header = 'infer'

        if parse_dates is True or (isinstance(parse_dates, (list, tuple)) is True and len(parse_dates) > 0):
            infer_datetime_format = True
        else:
            infer_datetime_format = False

        t = Timer()
        t.start()
        df = pd.read_csv(temp_file.path, sep=',',
                         header=header,
                         names=columns,
                         usecols=usecols,
                         index_col=index_col,
                         parse_dates=parse_dates,
                         infer_datetime_format=infer_datetime_format,
                         memory_map=True,
                         low_memory=False,
                         compression='infer'
                         )
        df.fillna(pd.np.nan, inplace=True)
        n_rows = df.shape[0]
        n_cols = len(df.columns)
        n_bytes = df.memory_usage(index=True, deep=False).sum()
        t.stop(n_items=int(n_rows),
               n_bytes=int(n_bytes),
               item_name='rows'
               )
        self.log.info("Returned DataFrame(cols={n_cols}, rows={n_rows}, size={bytes_str}) "
                      "in: '{duration}', '{bits_sec}', '{bytes_sec}', '{rows_sec}'".format(n_cols=n_cols,
                                                                                           n_rows=n_rows,
                                                                                           bytes_str=t.bytes_str,
                                                                                           duration=t.duration,
                                                                                           bits_sec=t.megabits_per_second_str,
                                                                                           bytes_sec=t.bytes_per_second_str,
                                                                                           rows_sec=t.items_per_second_str
                                                                                           ))

        return df


class FTPFolderMethods(FTPMethods):
    ftp: ftplib.FTP_TLS = NotImplemented
    path: str = NotImplemented

    directory: str = NotImplemented
    name: str = NotImplemented

    bytes: int = NotImplemented
    size: str = NotImplemented

    # md5: str = dataclasses.field(init=False)
    ctime: datetime.datetime = NotImplemented
    mtime: datetime.datetime = NotImplemented
    exists: bool = NotImplemented
    type: str = NotImplemented

    obj: dataclasses.InitVar[tuple] = NotImplemented
    classname: dataclasses.InitVar[str] = None
    file_classname : dataclasses.InitVar[str] = NotImplemented
    _classname: str = None
    _file_classname: str = None



    def __init__(self, ftp: ftplib.FTP_TLS, path: str, obj: tuple, classname: dataclasses.InitVar[str] = None):

        raise NotImplementedError

    def __repr__(self,
                 size: bool = True,
                 mtime: bool = False,
                 ctime: bool = False,
                 exists: bool = True
                 ):
        if self.exists is True:
            msgs = []
            if exists is True and self.exists is not None:
                msgs.append(f"exists={self.exists}")
            if size is True and self.size is not None:
                msgs.append(f"size='{self.size}'")
            if mtime is True and self.mtime is not None:
                msgs.append(f"mtime='{self.mtime}'")
            if ctime is True and self.ctime is not None:
                msgs.append(f"ctime='{self.ctime}'")

            if len(msgs) > 0:
                msgs_str = ", ".join(msgs)
                return f"{self.__class__.__name__}('{self.path}', {msgs_str})"
            else:
                return f"{self.__class__.__name__}('{self.path}')"
        else:
            return f"{self.__class__.__name__}('{self.path}', exists=False)"

            # return f"FTPFile('{self.path}', exists={self.exists})"

    def __str__(self,
                size: bool = True,
                mtime: bool = False,
                ctime: bool = False,
                exists: bool = True
                ):
        return self.__repr__(
                size=True,
                mtime=False,
                ctime=False,
                exists=True
                )


    def __post_init__(self, obj: tuple, classname: str, file_classname: str):
        self._classname = classname
        self.initialize(obj=obj, classname=classname, file_classname=file_classname)


    @property
    def classname(self):
        if self._classname is not None and isinstance(self._classname, str) and len(self._classname) > 0:
            return self._classname
        else:
            return self.__class__.__name__

    @property
    def file_classname(self):
        if self._file_classname is not None and isinstance(self._file_classname, str) and len(self._file_classname) > 0:
            return self._file_classname
        else:
            return 'FTPFile'


    def get_classname(self, classname: str = None):
        if classname is not None and isinstance(classname, str) and len(classname) > 0:
            return classname
        return None

    def set_classname(self, classname: str = None):
        self._classname = self.get_classname(classname)

    def rename_class(self, classname: str = None):
        og_classname = self.__class__.__name__
        new_classname = self.get_classname(classname)
        if new_classname is not None:
            self.__doc__ = (self.__class__.__name__ +
                            str(inspect.signature(self.__class__)).replace(' -> None', ''))
            self.__class__.__name__ = classname
            self.__class__.__qualname__ = classname



    def initialize(self, obj: tuple, classname: str = None, file_classname: str = None):
        parts = pathlib.PurePosixPath(self.path)
        self.directory = str(parts.parent)
        self.name = str(parts.name)
        self.extension = str(parts.suffix)
        self._classname = classname
        self._file_classname = file_classname

        if obj is None:
            obj = self._get_raw_object(ftp=self.ftp, path=self.path, type='dir')

        self.exists = False
        if obj is not None:
            assert obj[0] == self.name
            assert obj[1]['type'] == 'dir'

            assert 'modify' in obj[1]
            assert 'create' in obj[1]
            if obj[1]['modify'] != '19700131000000.000' and obj[1]['create'] != '19700131000000.000':
                self.exists = True
            self.bytes = int(obj[1]['size'])
            self.size = humansize(self.bytes)
            self.ctime = parse_time(obj[1]['create'])
            self.mtime = parse_time(obj[1]['modify'])
        self.rename_class(classname=classname)


    @classmethod
    def from_obj(cls,
                 ftp: ftplib.FTP_TLS,
                 obj: tuple,
                 parent_dir: str,
                 classname: str = None
                 ):
        """

        from an ftplib.File object
            ex. ('Sapphire End User Documentation',
                    {   'size': '7327748',
                        'modify': '20191018204619.000',
                        'create': '20180412221908.000',
                        'type': 'dir'
                    }
                )

        :param ftp:
        :param obj:
        :param parent_dir:
        :return:
        """

        assert obj[1]['type'] == 'dir'
        if parent_dir[-1] != "/":
            parent_dir = parent_dir + "/"
        path = parent_dir + obj[0]

        return cls(ftp=ftp,
                   path=path,
                   obj=obj,
                   classname=classname
                   )

    @classmethod
    def from_path(cls,
                  ftp: ftplib.FTP_TLS,
                  path: str,
                  classname: str = None
                  ):
        obj = cls._get_raw_object(ftp=ftp, path=path, type='dir', classname=classname)
        return cls(ftp=ftp, path=path, obj=obj, classname=classname)

    def to_obj(self, classname: str = None):
        obj = (self.name, {
                'create': datetime_to_ftptime(self.ctime),
                'modify': datetime_to_ftptime(self.mtime),
                'type':   'dir',
                'size':   str(self.bytes)
                }
                )

        if classname is not None and isinstance(classname, str) and len(classname) > 0:
            obj[1]['classname'] = classname
        if self.classname is not None:
            obj[1]['classname'] = self.classname
        return obj

    def refresh(self, obj: tuple):
        self.clear()
        self.__post_init__(obj=obj, classname=self.classname)

        return self

    def reset(self):
        return self.clear()

    def clear(self):
        self.exists = False
        self.size = humansize(0)
        self.bytes = 0
        self.ctime = parse_time('19700131000000.000')
        self.mtime = parse_time('19700131000000.000')
        return self

    def copy_obj(self):

        return FTPFolder(ftp=self.ftp, path=self.path, obj=self.to_obj(), classname=self.classname)

    def assert_exists(self):
        if self.exists is False:
            raise FTPFolderNotFoundError(self.path, classname=self.classname)

    def assert_not_exists(self):
        if self.exists is True:

            raise FTPFolderExistsError(self.path,  classname=self.classname)

    def file(self, name: str):
        if self.directory == "/" and self.path == "/":
            path = self.directory + name
        else:
            path = self.path + "/" + name
        return FTPFolder.from_path(ftp=self.ftp, path=path, classname=self.file_classname)

        # TO DO line 2379
    def folder(self, name: str):
        if self.directory == "/" and self.path == "/":
            path = self.directory + name
        else:
            path = self.path + "/" + name
        return FTPFolder.from_path(ftp=self.ftp, path=path, classname=self.classname)

    def files(self, path: str = None):
        if path is None:
            path = self.path
        return self._list_files(ftp=self.ftp, path=path, classname=self.file_classname)

    def folders(self, path: str = None):
        if path is None:
            path = self.path
        return self._list_dirs(ftp=self.ftp, path=path, classname=self.classname)

    def list(self,
             path: str = None
             ):
        if path is None:
            path = self.path
        return self._list_objs(ftp=self.ftp, path=path, file_classname=self.file_classname, folder_classname=self.classname)

    def upload_file(self,
                    path: str,
                    name: str = None
                    ):
        local_file = LocalFile(path)
        if name is None:
            name = local_file.name
        local_file.assert_exists()
        if self.directory == "/" and self.path == "/":
            remote_path = self.directory + name
        else:
            remote_path = self.path + "/" + name
        return self._upload_file(ftp=self.ftp,
                                 remote_path=remote_path,
                                 local_path=local_file.path,
                                 remote_file=self,
                                 classname=self.file_classname)

    def download_file(self,
                      name: str,
                      path: str = None
                      ):
        local_file = LocalFile(path)
        if self.directory == "/" and self.path == "/":
            remote_path = self.directory + name
        else:
            remote_path = self.path + "/" + name
        return self._download_file(ftp=self.ftp, local_path=local_file.path, remote_path=remote_path,
                                   classname=self.file_classname)

    def download_temp_file(self,
                           ftp: ftplib.FTP_TLS,
                           remote_path: str,
                           remote_file=None,
                           set_ctime=False,
                           set_mtime=True,
                           is_box: bool = False):

        return self._download_temp_file(ftp=self.ftp, remote_path=remote_path,
                                        classname=self.file_classname)


@dataclasses.dataclass(init=True, repr=False, eq=True, order=True, unsafe_hash=False, frozen=False)
class FTPFile(FTPFileMethods):
    ftp: ftplib.FTP_TLS = dataclasses.field(init=True)
    path: str = dataclasses.field(init=True)

    directory: str = dataclasses.field(init=False)
    name: str = dataclasses.field(init=False)
    extension: str = dataclasses.field(init=False)

    bytes: int = dataclasses.field(init=False, default=0)
    size: str = dataclasses.field(init=False, default=humansize(0))

    # md5: str = dataclasses.field(init=False)
    ctime: datetime.datetime = dataclasses.field(init=False, default=parse_time('19700131000000.000'))
    mtime: datetime.datetime = dataclasses.field(init=False, default=parse_time('19700131000000.000'))
    exists: bool = dataclasses.field(init=False, default=None)
    type: str = dataclasses.field(init=False, default='file')
    _md5: str = dataclasses.field(init=False, default=None)

    obj: dataclasses.InitVar[tuple] = NotImplemented
    classname: dataclasses.InitVar[str] = NotImplemented
    _classname: str = NotImplemented


    def __post_init__(self,
                      obj: tuple,
                      classname: str,
                      ):
        self.initialize(obj=obj, classname=classname)

@dataclasses.dataclass(init=True, repr=False, eq=True, order=True, unsafe_hash=False, frozen=False)
class FTPFolder(FTPFolderMethods):
    ftp: ftplib.FTP_TLS = dataclasses.field(init=True)
    path: str = dataclasses.field(init=True)

    directory: str = dataclasses.field(init=False)
    name: str = dataclasses.field(init=False)

    bytes: int = dataclasses.field(init=False, default=0)
    size: str = dataclasses.field(init=False, default=humansize(0))

    ctime: datetime.datetime = dataclasses.field(init=False, default=parse_time('19700131000000.000'))
    mtime: datetime.datetime = dataclasses.field(init=False, default=parse_time('19700131000000.000'))
    exists: bool = dataclasses.field(init=False, default=None)
    type: str = dataclasses.field(init=False, default='dir')

    obj: dataclasses.InitVar[tuple] = NotImplemented
    classname: dataclasses.InitVar[str] = None
    file_classname: dataclasses.InitVar[str] = NotImplemented
    _classname: str = None
    _file_classname: str = None

    def __post_init__(self,
                      obj,
                      classname,
                      file_classname
                      ):
        self.initialize(obj=obj, classname=classname, file_classname=file_classname)

    def __iter__(self):
        return iter(self.list(path=self.path))



@dataclasses.dataclass(init=True, repr=False, eq=True, order=True, unsafe_hash=False, frozen=False)
class BoxFTP(FTPClientMethods):
    # https://docs.python.org/3/library/ftplib.html#ftplib.FTP
    username: str = dataclasses.field(init=True)
    password: str = dataclasses.field(init=True)
    host: str = dataclasses.field(init=False, default='107.152.24.220')
    port: int = dataclasses.field(init=False, default=21)
    timeout: int = dataclasses.field(init=True, default=120)
    ftp: ftplib.FTP_TLS = dataclasses.field(init=False)

    file_classname = 'BoxFile'
    folder_classname = 'BoxFolder'

    def __post_init__(self):
        self.connect(ftp=None)



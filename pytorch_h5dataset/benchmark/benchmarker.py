import psutil
import numpy as np
import pandas as pd
class Benchmarker(object):

    def reset_benchmarker(self):
        self.__stats = []
        self.proc = psutil.Process()
        self.io_counters = self.proc.io_counters()
        self.cpu_times = self.proc.cpu_times()
        self.mem_stats = self.proc.memory_info()
        self.net_counters = psutil.net_io_counters(nowrap=True)

        self.bytes_recv_offset = self.net_counters.bytes_recv
        self.pack_recv_offset = self.net_counters.packets_recv

        self.io_count_offset = self.io_counters.read_count
        self.io_bytes_offset = self.io_counters.read_bytes

        self.rss_mem_offset = self.mem_stats.rss
        self.vms_mem_offset = self.mem_stats.vms

        self.cpu_system_offset = self.cpu_times.system
        self.cpu_user_offset = self.cpu_times.user


    def record_stats(self):
        io_counters = self.proc.io_counters()
        cpu_times = self.proc.cpu_times()
        mem_stats = self.proc.memory_info()
        net_counters = psutil.net_io_counters(nowrap=True)

        net_bytes_recv = net_counters.bytes_recv - self.bytes_recv_offset
        net_pack_recv  =  net_counters.packets_recv - self.pack_recv_offset

        io_count_read = io_counters.read_count - self.io_count_offset
        io_bytes_read = io_counters.read_bytes - self.io_bytes_offset

        rss_mem = mem_stats.rss - self.rss_mem_offset
        vms_mem = mem_stats.vms - self.vms_mem_offset

        cpu_system = cpu_times.system - self.cpu_system_offset
        cpu_user = cpu_times.user - self.cpu_user_offset


        self.__stats.append(
            {
                'proc_cpu_util': self.proc.cpu_percent(interval=None),
                'proc_cpu_count': len(psutil.cpu_percent(percpu=True)),
                'proc_mem_util_vms': self.proc.memory_percent('vms'),
                'proc_mem_util_rss': self.proc.memory_percent('rss'),
                'proc_mem_bytes_rss_acc': rss_mem,
                'proc_mem_bytes_vms_acc': vms_mem,
                'sys_net_io_bytes_recv_acc': net_bytes_recv,
                'sys_net_io_packets_recv_acc': net_pack_recv,
                'proc_disk_io_count_read_acc': io_count_read,
                'proc_disk_io_bytes_read_acc': io_bytes_read,
                'proc_cpu_time_system_acc': cpu_system,
                'proc_cpu_time_user_acc': cpu_user,
            })


    def decorate_iterator_class(_self, _iterator):
        class Iterator(object):
            def __init__(self, *args, **kwargs):
                self._iterator = _iterator(*args, **kwargs)
                self.iterator = None ## ugly TODO find a correct way

            def __next__(self):
                ret = next(self.iterator)
                _self.record_stats()
                return ret

            def __iter__(self):
                self.iterator = iter(self._iterator)
                return self

            def __len__(self):
                return len(self.iterator)

        return Iterator

    def decorate_iterator_func(self, _func):
        def iterate_function(*args, **kwargs):
            for ret in _func(*args, **kwargs):
                self.record_stats()
                yield ret

        return iterate_function

    def get_stats_df(self):
        self.df = pd.DataFrame.from_dict(self.__stats)
        self.df['proc_mem_bytes_rss'] = [0.]+(np.array(self.df['proc_mem_bytes_rss_acc'][1:])-np.array(self.df['proc_mem_bytes_rss_acc'][:-1])).tolist()
        self.df['proc_mem_bytes_vms'] = [0.]+(np.array(self.df['proc_mem_bytes_vms_acc'][1:])-np.array(self.df['proc_mem_bytes_vms_acc'][:-1])).tolist()
        self.df['sys_net_io_bytes_recv'] = [0.]+(np.array(self.df['sys_net_io_bytes_recv_acc'][1:])-np.array(self.df['sys_net_io_bytes_recv_acc'][:-1])).tolist()
        self.df['sys_net_io_packets_recv'] = [0.]+(np.array(self.df['sys_net_io_packets_recv_acc'][1:])-np.array(self.df['sys_net_io_packets_recv_acc'][:-1])).tolist()
        self.df['proc_disk_io_bytes_read'] = [0.]+(np.array(self.df['proc_disk_io_bytes_read_acc'][1:])-np.array(self.df['proc_disk_io_bytes_read_acc'][:-1])).tolist()
        self.df['proc_disk_io_count_read'] = [0.]+(np.array(self.df['proc_disk_io_count_read_acc'][1:])-np.array(self.df['proc_disk_io_count_read_acc'][:-1])).tolist()
        self.df['proc_cpu_time_system'] = [0.]+(np.array(self.df['proc_cpu_time_system_acc'][1:])-np.array(self.df['proc_cpu_time_system_acc'][:-1])).tolist()
        self.df['proc_cpu_time_user'] = [0.]+(np.array(self.df['proc_cpu_time_user_acc'][1:])-np.array(self.df['proc_cpu_time_user_acc'][:-1])).tolist()
        return self.df

    def stats(self):
        return self.__stats

    def __init__(self):
        self.proc = psutil.Process()
        self.reset_benchmarker()
        self.__stats = []
import time
import datetime


def datetime2timestamp(date_time_list):
    """
    convert the date time list to time stamp (epoch time)
    :param date_time_list:
    :return:
    """
    if len(date_time_list) < 6:
        exit("Error when convert the date time list, " + str(date_time_list))
    start_time = datetime.datetime(1970, 1, 1)
    timestamp = (datetime.datetime(date_time_list[0], date_time_list[1], date_time_list[2],
                                   date_time_list[3], date_time_list[4], date_time_list[5])
                 - start_time).total_seconds()
    return timestamp


def timestamp2datetime(timestamp):
    """
    convert the time stamp to date time list
    :param timestamp:
    :return:
    """
    temp_new_time = time.localtime(timestamp)
    year = temp_new_time.tm_year
    month = temp_new_time.tm_mon
    day = temp_new_time.tm_mday
    hour = temp_new_time.tm_hour
    minute = temp_new_time.tm_min
    second = temp_new_time.tm_sec
    date_time_list = [year, month, day, hour, minute, second]
    return date_time_list


def get_date_time_string(date_time_list):
    show_date_time_list = [str(date_time_list[0])]
    for element in date_time_list[1:]:
        if element < 10:
            show_date_time_list.append("0" + str(element))
        else:
            show_date_time_list.append(str(element))
    return '-'.join(show_date_time_list)


if __name__ == '__main__':
    a = timestamp2datetime(1464781722063/1000)
    a = get_date_time_string(a)
    print(a)

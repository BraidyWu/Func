import datetime

# day = ['2018-11-11']

def previous_day(day):
    previous = {}
    for i in day.keys():
        theday = datetime.date(*map(int, day.split('-')))
        prevday = theday - datetime.timedelta(days=1)
        prevday = prevday.strftime('%Y-%m-%d')
        previous[i] = prevday
    return previous


def after_day(day):
    after = {}
    for i in day.keys():
        theday = datetime.date(*map(int, day.split('-')))
        afterday = theday - datetime.timedelta(days=1)
        afterday = afterday.strftime('%Y-%m-%d')
        after[i] = afterday
    return after


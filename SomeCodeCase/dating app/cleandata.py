import re

def cleandata(use_information):
    social_time_day = use_information['Ideally, I would like to socialize with my partner 1:']
    social_time_hour = use_information['Ideally, I would like to socialize with my partner 2:']
    social_time_day = max(re.findall('\d+', social_time_day))
    social_time_hour = max(re.findall('\d+', social_time_hour))
    use_information['Ideally, I would like to socialize with my partner 1:'] = social_time_day
    use_information['Ideally, I would like to socialize with my partner 2:'] = social_time_hour
    

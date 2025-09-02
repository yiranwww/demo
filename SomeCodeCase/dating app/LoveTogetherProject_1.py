# To find the min difference bwteen the two person and to give higer socre.
# we can adjust the weighted items if the area is more important than the others.
# Overall is a basic idea on how to give scores, there are many details to be optimized.

import openpyxl
import re




def cleandata(use_information):
    social_time_day = use_information['Ideally, I would like to socialize with my partner 1:']
    social_time_hour = use_information['Ideally, I would like to socialize with my partner 2:']
    social_time_day = int(max(re.findall('\d+', social_time_day)))
    social_time_hour = int(max(re.findall('\d+', social_time_hour)))
    use_information['Ideally, I would like to socialize with my partner 1:'] = social_time_day
    use_information['Ideally, I would like to socialize with my partner 2:'] = social_time_hour

    childrenProcess = use_information['Do you have children?']
    if childrenProcess == 'Yes':
        use_information['Do you have children?'] = 10
    else:
        use_information['Do you have children?'] = 1
    
    childTimecur = use_information['Currently, I spend \_\_\_\_\_\_ with my children.']
    if len(childTimecur) != 0:
        childTimecur = int(max(re.findall('\d+', childTimecur)))
        use_information['Currently, I spend \_\_\_\_\_\_ with my children.'] = childTimecur
    else:
        use_information['Currently, I spend \_\_\_\_\_\_ with my children.'] = None

    childIdeal = use_information['Ideally, I would spend \_\_\_\_\_\_ with my children.']
    if len(childIdeal) != 0:
        childIdeal = int(max(re.findall('\d+', childIdeal)))
        use_information['Ideally, I would spend \_\_\_\_\_\_ with my children.'] = childIdeal
    else:
        use_information['Ideally, I would spend \_\_\_\_\_\_ with my children.'] = None

    haveChild = use_information['Are you interested in having children?']
    if type(haveChild) == int:
        use_information['Are you interested in having children?'] = haveChild
    else:
        use_information['Are you interested in having children?'] = None


    HaveChildToge= use_information['I can be in a relationship with a person who already has a child (or children).']
    if HaveChildToge == 'Yes':
        use_information['I can be in a relationship with a person who already has a child (or children).'] = 10
    else:
        use_information['I can be in a relationship with a person who already has a child (or children).'] = 0

    careInlaw = use_information['It is standard for me and my partner to take care of my in-laws.']
    if careInlaw == 'Yes':
        use_information['It is standard for me and my partner to take care of my in-laws.'] = 10
    else:
        use_information['It is standard for me and my partner to take care of my in-laws.'] = 1

    weekSpend = use_information['During the average week, I spend (in USD):']
    minSpend, maxSpend = min(re.findall('\d+', weekSpend)), max(re.findall('\d+', weekSpend))
    weekSpend = int((int(minSpend) + int(maxSpend)) / 100)
    use_information['During the average week, I spend (in USD):'] = weekSpend

    bankShare = use_information['If you and your partner were to get married, do you want a joint bank account with your partner?']
    if bankShare == 'Yes, but I also want to have separate bank accounts.':
        use_information['If you and your partner were to get married, do you want a joint bank account with your partner?'] = 5
    elif bankShare == 'Yes, I want to share everything with my partner.':
        use_information['If you and your partner were to get married, do you want a joint bank account with your partner?'] = 10
    else:
        use_information['If you and your partner were to get married, do you want a joint bank account with your partner?'] = 1

    sexTime = use_information['Ideally, I would like to have sexual intercourse:']
    sexTime = int(max(re.findall('\d+', sexTime)))
    use_information['Ideally, I would like to have sexual intercourse:'] = sexTime

    afterDateSex = use_information['At what point are you willing to have sexual intercourse?']
    if 'months' in afterDateSex:
        use_information['At what point are you willing to have sexual intercourse?'] = 10
    else:
        use_information['At what point are you willing to have sexual intercourse?'] = 5
    
    workTime = use_information['On average, I typically work:']
    workTime = int(max(re.findall('\d+', workTime)))
    use_information['On average, I typically work:'] = workTime

    ageRange = use_information['What is your age range?']
    if ageRange == '35-44':
        use_information['What is your age range?'] = 3
    elif ageRange == '25-34':
        use_information['What is your age range?'] = 2
    elif ageRange == '45-54':
        use_information['What is your age range?'] = 4
    elif ageRange == '55-64':
        use_information['What is your age range?'] = 6
    else:
        use_information['What is your age range?'] = 10

    
    

def datetwoPeople(user1, user2):
    final_score = 0
    gender_key = 'How would you describe your gender?'
    if user1[gender_key] == user2[gender_key]:
        return 0

    num_key = ['If my partner suddenly said, "We are going on a date today," I would be perfectly fine with the late notice.', 
            'I worry if my partner does not respond to my text and/or call after 3 hours.', 
            'I am a sarcastic person.', 
            'During regular conversations, I make sharp remarks.', 
            'I would rather have a partner that directly communicates their frustrations with me.', 
            'Even if I completely understood my partners words, I would become upset, frustrated, and/or disheartened if their communication came across as harsh.',
            # 'I would describe myself as an extroverted person.', 
            # 'Its Saturday morning and you are very tired from yesterdays workday. However, your partner asked if you wanted to have brunch today. How likely are you to accept this offer?', 
            'Ideally, I would like to socialize with my partner 1:', 
            'Ideally, I would like to socialize with my partner 2:', 
            'When angered, how likely are you to blurt out words?', 
            'You just finished having an argument with your partner. How likely are you to give your partner the silent treatment?',
            'It is hard for me to admit that I am/was wrong during an argument, debate, etc.',
            'It is difficult for me to apologize for my wrongdoings.', 
            'I wouldnt be able to stay in a long-distance relationship for more than a year.',
            'You and your partner are married. Your partner has notified you that they want to move to a different city because they want to advance their career. They also want you to live with them in this new city. How likely are you to accept this offer?', 
            'Do you have children?', 
            'Currently, I spend \_\_\_\_\_\_ with my children.', 
            'Are you interested in having children?', 
            'Ideally, I would spend \_\_\_\_\_\_ with my children.', 
            'I can be in a relationship with a person who already has a child (or children).', 
            'I am fine with me and/or my partner shouting at our kids.', 
            'Religion is very important to me.', 
            'I want my partner to practice the same religion and sect as me.', 
            'It is standard for me and my partner to take care of my in-laws.', 
            'If my partner and my parents were facing some kind of conflict, I would side with my partner.', 
            'If necessary, I would be fine if I had to live with my partners in-laws.', 
            'I would describe myself as more of a spender (someone who likes to splurge) than a saver.', 
            'During the average week, I spend (in USD):', 
            'If you and your partner were to get married, do you want a joint bank account with your partner?', 
            'I desire a partner that shares the same educational level as me.',
            'If I am focused on post-secondary education, I will not be able to stay committed to a relationship.', 
            'I am willing to sacrifice hang-out time, physical affection, etc. if my partner is pursuing post-secondary education.', 
            'I am willing to sacrifice some post-secondary activities, projects, and/or opportunities for my relationship.', 
            'I think sex is very important for a relationship.', 
            'Your partner is asking for more sexual interactions during the average 7-day week. However, you feel that your weekly sexual interactions are more than adequate. How likely are you to grant your partners wish for more sexual interactions?',
            'Your partner is asking for less sexual interactions during the average 7-day week. However, you feel that your weekly sexual interactions are more than adequate. How likely are you to grant your partners wish for less sexual interactions?', 
            'Ideally, I would like to have sexual intercourse:', 
            'At what point are you willing to have sexual intercourse?', 
            'I keep my living space very clean.',
            'I cook at home more than eating out.',
            'I am on-time for every commitment.', 
            'On average, I typically work:', 
            'I am willing to adjust some of my habits if it strengthens me and my partners relationship.', 
            'I like to eat very healthy food.', 
            'I like to workout.', 
            'I like to wear clothes that align with fashion trends.', 
            'I enjoy going to parties.', 
            'I am very flexible when it comes to changing my long-term goals.', 
            'When it comes to love, I would identify myself as more of a romantic person than a practical person.', 
            'What is your age range?',

            ]
            
    for cur_key in num_key:
        user1_cur = user1[cur_key]
        user2_cur = user2[cur_key]
        if user1_cur != None and user2_cur != None:
            user_diff = 10-abs(user1_cur - user2_cur)
        else:
            user_diff = 0
        final_score += user_diff


    text_key = [
        'What are four interests that you have?', 
        'What are four personal values that you have?',
        'Rate your expressions of love.',
        'What is your current religion, if any?',
        'What religious rituals do you practice?', 
        'If your relationship is beginning to lose its sparks, how would you try to rekindle the sparks in your relationship?'

    ]

    for cur_key in text_key:
        user1_cur = user1[cur_key].split(',')
        user2_cur = user2[cur_key].split(',')
        count=0 
        for a in user1_cur:
            if(a in user2_cur):
                count+=1
        final_score += count 



    weighted_key = [
        'I would describe myself as an extroverted person.', 
        'Its Saturday morning and you are very tired from yesterdays workday. However, your partner asked if you wanted to have brunch today. How likely are you to accept this offer?'
            ]
    
    for cur_key in weighted_key:
        user1_cur = user1[cur_key]
        user2_cur = user2[cur_key]
        if user1_cur != None and user2_cur != None:
            user_diff = (10-abs(user1_cur - user2_cur)) * 3
        else:
            user_diff = 0
        final_score += user_diff

    return final_score
    

    

## Load database
wb = openpyxl.load_workbook(filename = './alogrithmtestmodified.xlsx')
sheet = wb.worksheets[0]

## Load information for each person
keyvalue = sheet["A"]
user_1 = sheet['B']
user_2 = sheet['C']
user_3 = sheet['E']
user_4 = sheet['F']

user_1_information={}
user_2_information={}
user_3_information={}
user_4_information={}
for i in range(len(keyvalue)):
    user_1_information[keyvalue[i].value] = user_1[i].value
    user_2_information[keyvalue[i].value] = user_2[i].value
    user_3_information[keyvalue[i].value] = user_3[i].value
    user_4_information[keyvalue[i].value] = user_4[i].value
    # user_1_information.add(keyvalue[i].value, user_1[i].value)



# clean the data. 
cleandata(user_1_information)
cleandata(user_2_information)

cleandata(user_3_information)
cleandata(user_4_information) # use the clean function for person you only want to see



final_score = datetwoPeople(user_1_information, user_2_information) # we can also save multiple results and rank them
print(final_score)



# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 17:49:34 2023

@author: jacki
"""
import tiktoken
import os
import openai
import json
import time
import argparse
argparser=argparse.ArgumentParser()
argparser.add_argument('--input', type=str,help='input summary json file')
argparser.add_argument('--dataset',type=str,help='dataset where the input summary is generated')
args = argparser.parse_args()
openai.api_key = '' #fill in the openai api key
if args.dataset=='amazon':
    acu_dir='acu_sample/amazon'
elif args.dataset=='mitweet':
    acu_dir='acu_sample/tweet'
elif args.dataset=='semeval':
    acu_dir='acu_sample/tweet'
elif args.dataset=='news_stance':
    acu_dir='acu_sample/news_stance'
elif args.dataset=='article_bias':
    acu_dir='acu_sample/article_bias'
else:
    print('dataset is not defined')
    assert 0==1
samples=[]
for i in os.listdir(acu_dir):
    sample=[]
    with open(os.path.join(acu_dir,i),'r') as file:
        for j in file:
            sample.append(j.strip())
    sample[0]=sample[0].replace('\t',' ')
    sample[0]="Summary: "+sample[0]
    samples.append(sample)
#samples=samples[0:2]
#assert 0==1
if args.dataset=='amazon':
    prompt="Each atomic content unit contain an atomic fact and does not need further split for the purpose of reducing ambiguity in human evaluation. Avoid using reported speech in the atomic content units. Use passive voice or appropriate pronouns or adjectives to avoid mentioning 'customer', product names,'review' or 'comment' in the atomic content units. An atomic content unit should not contain compound suject or compound object. If a sentence contain compound objects or compound subjects, extract an individual atomic content unit for each object or subject. Therefore, different atomic content units can share some similar contents. Below are some examples of extracted atomic content units of corresponding summaries.\n\n"
    for i in samples:
        media='\n'.join(i)+'\n\n'
        prompt=prompt+media
    prompt+="Extract atomic content units for the following summary following the form of the above examples.\n\n"
elif args.dataset=='mitweet':
    prompt="Each atomic content unit contain an atomic fact and does not need further split for the purpose of reducing ambiguity in human evaluation. Avoid using reported speech in the atomic content units. Use passive voice or appropriate pronouns or adjectives to avoid mentioning 'tweet' in the atomic content units. An atomic content unit should not contain compound suject or compound object. If a sentence contain compound objects or compound subjects, extract an individual atomic content unit for each object or subject. Therefore, different atomic content units can share some similar contents. Avoid using 'a mix of A and B' in the atomic content units. Divide it into two atomic content units. Below are some examples of extracted atomic content units of corresponding summaries. \n\n"
    for i in samples:
        media='\n'.join(i)+'\n\n'
        prompt=prompt+media
    prompt+="Extract atomic content units for the following summary following the form of the above examples.\n\n"
elif args.dataset=='semeval':
    prompt="Each atomic content unit contain an atomic fact and does not need further split for the purpose of reducing ambiguity in human evaluation. Avoid using reported speech in the atomic content units. Use passive voice or appropriate pronouns to avoid mentioning 'tweet', 'supporters', 'some', 'others', 'skeptics', 'crtics' or their synonyms in the atomic content units. An atomic content unit should not contain compound suject or compound object. If a sentence contain compound objects or compound subjects, extract an individual atomic content unit for each object or subject. Therefore, different atomic content units can share some similar contents. Avoid using 'a mix of A and B' in the atomic content units. Divide it into two atomic content units. Below are some examples of extracted atomic content units of corresponding summaries. \n\n"
    for i in samples:
        media='\n'.join(i)+'\n\n'
        prompt=prompt+media
    prompt+="Extract atomic content units for the following summary following the form of the above examples.\n\n"
elif args.dataset=='news_stance':
    prompt="Each atomic content unit contain an atomic fact and does not need further split for the purpose of reducing ambiguity in human evaluation. An atomic content unit should not be too long. If a summary sentence contain compound objects or compound subjects, extract an atomic content unit for each object or subject. Therefore, different atomic content units can share some similar contents. Do not use reported speech for any atomic content units. Replace 'documents', 'sources', 'reports', 'others', 'evidences' or their combinations in each extracted atomic content unit with 'it'.  Below are some examples of extracted atomic content units from corresponding summaries. \n\n"
    for i in samples:
        media='\n'.join(i)+'\n\n'
        prompt=prompt+media
    prompt+="Extract atomic content units for the following summary following the form of the above examples.\n\n"
elif args.dataset=='article_bias':
    prompt="Each atomic content unit contain an atomic fact and does not need further split for the purpose of reducing ambiguity in human evaluation.  If a sentence contain compound objects or compound subjects, extract an atomic content unit for each object or subject. Therefore, different atomic content units can share some similar contents. Do not use reported speech for any atomic content units. For each atomic content unit, replace 'the articles', 'the news articles'， 'the media', 'the narrative' or their synonyms at the begining with 'it'.  Below are some examples of extracted atomic content units of corresponding summaries.  \n\n"
    for i in samples:
        media='\n'.join(i)+'\n\n'
        prompt=prompt+media
    prompt+="Extract atomic content units for the following summary following the form of the above examples.\n\n"
prompts={}
with open(args.input+'.json','r') as file:
    #media=file.readlines()
    result=json.load(file)
#result={i:result[i]['gpt3.5'] for i in result.keys()}
for i in result.keys():
    prompts[i]= prompt+'Summary: '+result[i]+'\nExtracted Atomic Content Units:'
#assert 0==1
result1={}
for i in prompts.keys():
    #print(i)
    success=False
    while not success:
        try:
            #print(i)
            message=[{'role':'system','content':'You are a helpful assistant.'}]
            message.append({'role':'user','content':prompts[i]})
            response = openai.ChatCompletion.create(
            temperature=1.0,
            model="gpt-3.5-turbo-0125",
            messages=message,max_tokens=384)
            result1[i]=response['choices'][0]['message']['content']
            success=True
        except:
            time.sleep(1)
            print(i+"error")
    break
            
result={}
num_acu=0
for i in result1.keys():
    media=result1[i].strip().split('\n')
    media_result=[]
    for k in range(len(media)):
       media_token=media[k].split(' ')
       found_numeric=False
       for l in media_token[0]:
           if l.isnumeric():
               found_numeric=True
               break
       if found_numeric:
           media_result.append(media[k][len(media_token[0]):].strip())
       else:
           print(media[k])
    if args.dataset=='amazon':
        for j in range(len(media_result)):
            if 'some customers' in media_result[j].lower():
                #media_result[j]='They'+media_result[j][14:]
                media_result[j]=media_result[j].replace('customers','')
                media_result[j]=media_result[j].replace('Customers','')
                media_result[j]=media_result[j].replace('  ',' ').strip()
                assert len(media_result[j])>5
                print(i)
                print(media_result[j].strip())
            elif 'customers' in media_result[j].lower():
                #media_result[j]='They'+media_result[j][9:]
                media_result[j]=media_result[j].replace('customers','they')
                media_result[j]=media_result[j].replace('Customers','They')
                media_result[j]=media_result[j].replace('  ',' ').strip()
                assert len(media_result[j])>5
                print(i)
                print(media_result[j].strip())
            if 'some users' in media_result[j].lower():
                #media_result[j]='They'+media_result[j][14:]
                media_result[j]=media_result[j].replace('users','')
                media_result[j]=media_result[j].replace('users','')
                media_result[j]=media_result[j].replace('  ',' ').strip()
                assert len(media_result[j])>5
                print(i)
                print(media_result[j].strip())
    elif args.dataset=='article_bias':
        for k in range(len(media_result)):
            if media_result[k].lower().startswith('the articles'):
                media_result[k]='It'+media_result[k][12:]
                print(media_result[k])
            if media_result[k].lower().startswith('the news articles'):
                media_result[k]='It'+media_result[k][17:]
                print(media_result[k])
    elif args.dataset=='news_stance':
        for k in range(len(media_result)):
            if media_result[k].lower().startswith('the articles'):
                media_result[k]='It'+media_result[k][12:]
                print(media_result[k])
            if media_result[k].lower().startswith('the news articles'):
                media_result[k]='It'+media_result[k][17:]
                print(media_result[k])
            if media_result[k].lower().startswith('the documents'):
                media_result[k]='It'+media_result[k][13:]
                print(media_result[k])
            if media_result[k].lower().startswith('the news documents'):
                media_result[k]='It'+media_result[k][18:]
                print(media_result[k])
            if media_result[k].lower().startswith('documents'):
                media_result[k]='It'+media_result[k][9:]
                print(media_result[k])
    elif args.dataset=='mitweet' or args.dataset=='semeval':
        for k in range(len(media_result)):
            if media_result[k].lower().startswith('the tweets'):
                media_result[k]='It'+media_result[k][len('the tweets'):]
                print(media_result[k])
            if media_result[k].lower().startswith('tweets'):
                media_result[k]='It'+media_result[k][len('tweets'):]
                print(media_result[k])
    #result[i]=media_result
    #num_acu+=len(result[i])
    if len(media_result)>2:
        result[i]=media_result
        num_acu+=len(result[i])
    else:
        print(media_result)
        print(i)
with open(args.input+'_acu_preprocessed.json','w') as file:
    json.dump(result,file)
                

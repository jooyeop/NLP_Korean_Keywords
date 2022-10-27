import os
os.environ['JAVA_HOME'] = r'D:\jdk-19.0.1\bin'

import numpy as np
import itertools

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

doc = """
드론 활용 범위도 점차 확대되고 있다. 최근에는 미세먼지 관리에 드론이 활용되고 있다.
서울시는 '미세먼지 계절관리제' 기간인 지난달부터 오는 3월까지 4개월간 드론에 측정장치를 달아 미세먼지 집중 관리를 실시하고 있다.
드론은 산업단지와 사업장 밀집지역을 날아다니며 미세먼지 배출 수치를 점검하고, 현장 모습을 영상으로 담는다.
영상을 통해 미세먼지 방지 시설을 제대로 가동하지 않는 업체와 무허가 시설에 대한 단속이 한층 수월해질 전망이다.
드론 활용에 가장 적극적인 소방청은 광범위하고 복합적인 재난 대응 차원에서 드론과 관련 전문인력 보강을 꾸준히 이어가고 있다.
지난해 말 기준 소방청이 보유한 드론은 총 304대, 드론 조종 자격증을 갖춘 소방대원의 경우 1,860명이다.
이 중 실기평가지도 자격증까지 갖춘 ‘드론 전문가’ 21명도 배치돼 있다.
소방청 관계자는 "소방드론은 재난현장에서 영상정보를 수집, 산악ㆍ수난 사고 시 인명수색·구조활동,
유독가스·폭발사고 시 대원안전 확보 등에 활용된다"며
"향후 화재진압, 인명구조 등에도 드론을 활용하기 위해 연구개발(R&D)을 하고 있다"고 말했다.
"""

okt = Okt()

# doc 핵심 단어
tokenized_doc = okt.pos(doc)

tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])
print('핵심 키워드 :',tokenized_nouns)
"""
형태소 분석 모듈
konlpy와 kiwipiepy를 지원
"""
import logging
from typing import List, Union, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MorphologicalAnalyzer:
    """형태소 분석기 래퍼 클래스"""
    
    def __init__(self, analyzer_type: str = "kiwi"):
        """
        Args:
            analyzer_type: 'kiwi', 'kkma', 'komoran', 'mecab', 'okt' 중 선택
        """
        self.analyzer_type = analyzer_type.lower()
        self.analyzer = None
        self._initialize_analyzer()
    
    def _initialize_analyzer(self):
        """형태소 분석기 초기화"""
        try:
            if self.analyzer_type == "kiwi":
                from kiwipiepy import Kiwi
                self.analyzer = Kiwi()
                logger.info("Kiwi 형태소 분석기 초기화 완료")
            
            elif self.analyzer_type == "kkma":
                from konlpy.tag import Kkma
                self.analyzer = Kkma()
                logger.info("Kkma 형태소 분석기 초기화 완료")
            
            elif self.analyzer_type == "komoran":
                from konlpy.tag import Komoran
                self.analyzer = Komoran()
                logger.info("Komoran 형태소 분석기 초기화 완료")
            
            elif self.analyzer_type == "mecab":
                from konlpy.tag import Mecab
                self.analyzer = Mecab()
                logger.info("Mecab 형태소 분석기 초기화 완료")
            
            elif self.analyzer_type == "okt":
                from konlpy.tag import Okt
                self.analyzer = Okt()
                logger.info("Okt 형태소 분석기 초기화 완료")
            
            else:
                raise ValueError(f"지원하지 않는 분석기 타입: {self.analyzer_type}")
        
        except ImportError as e:
            logger.error(f"형태소 분석기 라이브러리 임포트 실패: {e}")
            logger.info("기본적으로 Kiwi를 사용합니다. pip install kiwipiepy로 설치하세요.")
            try:
                from kiwipiepy import Kiwi
                self.analyzer = Kiwi()
                self.analyzer_type = "kiwi"
            except:
                raise ImportError("형태소 분석기 라이브러리를 설치해주세요: pip install konlpy kiwipiepy")
    
    def analyze(self, text: str, pos_filter: Optional[List[str]] = None) -> List[Union[str, tuple]]:
        """
        형태소 분석 수행
        
        Args:
            text: 분석할 텍스트
            pos_filter: 포함할 품사 리스트 (None이면 모든 품사)
                        예: ['Noun', 'Verb', 'Adjective']
        
        Returns:
            형태소 분석 결과 리스트
        """
        if not text or not text.strip():
            return []
        
        try:
            if self.analyzer_type == "kiwi":
                result = self.analyzer.analyze(text)
                # Kiwi 결과를 (형태소, 품사) 튜플 리스트로 변환
                morphemes = []
                for word in result[0][0]:
                    morphemes.append((word.form, word.tag))
                
                if pos_filter:
                    morphemes = [(m, p) for m, p in morphemes if p in pos_filter]
                
                return morphemes
            
            else:
                # konlpy 분석기들
                result = self.analyzer.pos(text)
                
                if pos_filter:
                    result = [(m, p) for m, p in result if p in pos_filter]
                
                return result
        
        except Exception as e:
            logger.error(f"형태소 분석 오류: {e}")
            return []
    
    def extract_nouns(self, text: str) -> List[str]:
        """명사만 추출"""
        if self.analyzer_type == "kiwi":
            result = self.analyzer.analyze(text)
            nouns = [word.form for word in result[0][0] if word.tag.startswith('N')]
        else:
            nouns = self.analyzer.nouns(text)
        return nouns
    
    def extract_keywords(self, text: str, min_length: int = 2) -> List[str]:
        """
        키워드 추출 (명사, 동사, 형용사)
        
        Args:
            text: 분석할 텍스트
            min_length: 최소 글자 수
        
        Returns:
            키워드 리스트
        """
        if self.analyzer_type == "kiwi":
            pos_filter = ['NNG', 'NNP', 'NNB', 'VV', 'VA', 'VX', 'VA']
        else:
            pos_filter = ['Noun', 'Verb', 'Adjective']
        
        morphemes = self.analyze(text, pos_filter=pos_filter)
        keywords = [m for m, p in morphemes if len(m) >= min_length]
        
        return keywords
    
    def tokenize(self, text: str) -> List[str]:
        """
        간단한 토큰화 (형태소 단위)
        
        Args:
            text: 토큰화할 텍스트
        
        Returns:
            토큰 리스트
        """
        morphemes = self.analyze(text)
        return [m for m, p in morphemes]
    
    def analyze_batch(self, texts: List[str], pos_filter: Optional[List[str]] = None) -> List[List[Union[str, tuple]]]:
        """
        여러 텍스트에 대한 형태소 분석 (배치 처리)
        
        Args:
            texts: 분석할 텍스트 리스트
            pos_filter: 포함할 품사 리스트
        
        Returns:
            형태소 분석 결과 리스트의 리스트
        """
        return [self.analyze(text, pos_filter) for text in texts]


import tkinter as tk
from tkinter import ttk
import pandas as pd
import edge_tts
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

class VocaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("단어 학습기")
        self.root.geometry("800x600")
        self.root.configure(bg='#2C3E50')  # 어두운 배경색
        
        # 엑셀 파일 읽기
        self.df = pd.read_excel('voca1000.xlsx')
        self.current_index = 0
        self.is_playing = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # 재생 속도 설정
        self.playback_speed = 1.8
        
        # UI 구성
        self.create_widgets()
        
        # 첫 번째 단어 표시
        self.show_word()
        
    def create_widgets(self):
        # 메인 프레임
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # 속도 조절 프레임
        speed_frame = ttk.Frame(main_frame)
        speed_frame.pack(pady=10)
        
        # 속도 레이블
        speed_label = tk.Label(
            speed_frame,
            text="재생 속도:",
            font=("Arial", 12),
            bg='#2C3E50',
            fg='white'
        )
        speed_label.pack(side=tk.LEFT, padx=5)
        
        # 속도 표시 레이블 (먼저 생성)
        self.speed_value_label = tk.Label(
            speed_frame,
            text="1.8x",
            font=("Arial", 12),
            bg='#2C3E50',
            fg='white'
        )
        self.speed_value_label.pack(side=tk.LEFT, padx=5)
        
        # 속도 슬라이더
        self.speed_scale = ttk.Scale(
            speed_frame,
            from_=0.5,
            to=4.0,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.update_speed
        )
        self.speed_scale.set(1.8)
        self.speed_scale.pack(side=tk.LEFT, padx=5)
        
        # 한국어 의미 표시
        self.kor_label = tk.Label(
            main_frame,
            text="",
            font=("Arial", 36, "bold"),
            bg='#2C3E50',
            fg='white',
            pady=20
        )
        self.kor_label.pack(pady=20)
        
        # 영어 단어 표시
        self.eng_label = tk.Label(
            main_frame,
            text="",
            font=("Arial", 28),
            bg='#2C3E50',
            fg='#3498DB',
            pady=20
        )
        self.eng_label.pack(pady=20)
        
        # 중국어 의미 표시
        self.chi_label = tk.Label(
            main_frame,
            text="",
            font=("Arial", 28),
            bg='#2C3E50',
            fg='#E74C3C',
            pady=20
        )
        self.chi_label.pack(pady=20)
        
        # 버튼 프레임
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=30)
        
        # 버튼 스타일 설정
        style = ttk.Style()
        style.configure('Custom.TButton', 
                       font=('Arial', 28, 'bold'),
                       padding=15)
        
        # 버튼 크기 설정
        button_width = 6
        
        # 이전 버튼
        self.prev_btn = ttk.Button(
            button_frame,
            text="이전",
            command=self.prev_word,
            style='Custom.TButton',
            width=button_width
        )
        self.prev_btn.pack(side=tk.LEFT, padx=10)
        
        # 다음 버튼
        self.next_btn = ttk.Button(
            button_frame,
            text="다음",
            command=self.next_word,
            style='Custom.TButton',
            width=button_width
        )
        self.next_btn.pack(side=tk.LEFT, padx=10)
        
        # 발음 버튼
        self.speak_btn = ttk.Button(
            button_frame,
            text="발음",
            command=self.speak_word,
            style='Custom.TButton',
            width=button_width
        )
        self.speak_btn.pack(side=tk.LEFT, padx=10)
        
        # 자동 재생 버튼
        self.auto_btn = ttk.Button(
            button_frame,
            text="자동 재생",
            command=self.toggle_auto_play,
            style='Custom.TButton',
            width=button_width
        )
        self.auto_btn.pack(side=tk.LEFT, padx=10)
        
    def update_speed(self, value):
        self.playback_speed = float(value)
        self.speed_value_label.config(text=f"{self.playback_speed:.1f}x")
        
    def show_word(self):
        if 0 <= self.current_index < len(self.df):
            self.kor_label.config(text=self.df.iloc[self.current_index, 1])
            self.eng_label.config(text=self.df.iloc[self.current_index, 0])
            self.chi_label.config(text=self.df.iloc[self.current_index, 2])
            
    def next_word(self):
        if self.current_index < len(self.df) - 1:
            self.current_index += 1
            self.show_word()
            
    def prev_word(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_word()
            
    async def speak(self, text, voice='en-US-JennyNeural'):
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save("temp.mp3")
            
            # 음성 재생 속도 조절
            if self.playback_speed != 1.0:
                # ffmpeg를 사용하여 음성 속도 조절
                os.system(f"/opt/homebrew/bin/ffmpeg -i temp.mp3 -filter:a atempo={self.playback_speed} -y temp_speed.mp3")
                os.system("afplay temp_speed.mp3")
                os.remove("temp_speed.mp3")
            else:
                os.system("afplay temp.mp3")
                
            os.remove("temp.mp3")
        except Exception as e:
            print(f"음성 재생 중 오류 발생: {e}")
            
    def speak_word(self):
        if 0 <= self.current_index < len(self.df):
            word = self.df.iloc[self.current_index, 0]
            asyncio.run(self.speak(word))
            
    def toggle_auto_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.auto_btn.config(text="정지")
            self.executor.submit(self.run_auto_play)
        else:
            self.auto_btn.config(text="자동 재생")
            
    def run_auto_play(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.auto_play_sequence())
        loop.close()
            
    async def auto_play_sequence(self):
        while self.is_playing and self.current_index < len(self.df):
            try:
                # 현재 단어 표시
                self.root.after(0, self.show_word)
                
                # 대기 시간을 재생 속도에 맞게 조절
                wait_time = 0.1 / self.playback_speed
                await asyncio.sleep(wait_time)
                
                # 한국어 발음
                kor_word = self.df.iloc[self.current_index, 1]
                await self.speak(kor_word, 'ko-KR-SunHiNeural')
                
                # 대기 시간을 재생 속도에 맞게 조절
                wait_time = 0.5 / self.playback_speed
                await asyncio.sleep(wait_time)
                
                # 영어 단어 발음
                eng_word = self.df.iloc[self.current_index, 0]
                await self.speak(eng_word, 'en-US-JennyNeural')
                
                # 다음 단어로 이동
                if self.current_index < len(self.df) - 1:
                    self.current_index += 1
                    # 대기 시간을 재생 속도에 맞게 조절
                    wait_time = 0.5 / self.playback_speed
                    await asyncio.sleep(wait_time)
                else:
                    self.is_playing = False
                    self.root.after(0, lambda: self.auto_btn.config(text="자동 재생"))
            except Exception as e:
                print(f"자동 재생 중 오류 발생: {e}")
                self.is_playing = False
                self.root.after(0, lambda: self.auto_btn.config(text="자동 재생"))
                break

if __name__ == "__main__":
    root = tk.Tk()
    app = VocaApp(root)
    root.mainloop()

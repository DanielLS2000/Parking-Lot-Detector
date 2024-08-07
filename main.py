import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

def calculate_histograms(image):
    histograms = []
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        histograms.append(hist)
    return histograms

def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    scores = []
    for h1, h2 in zip(hist1, hist2):
        score = cv2.compareHist(h1, h2, method)
        scores.append(score)
    return np.mean(scores)

class Vaga:
    def __init__(self, coords, ocupada, hist):
        self.coords = coords
        self.ocupada_inicialmente = ocupada
        self.ocupada = ocupada
        self.initial_hist = hist
        self.occupation_history = [ocupada]

    def register_occupation(self):
        if len(self.occupation_history) < 30:
            self.occupation_history.append(self.ocupada)
        else:
            self.occupation_history.pop(0)
            self.occupation_history.append(self.ocupada)


class VideoApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Analisador de Vagas")
        self.screen_width = self.master.winfo_screenwidth()
        self.screen_height = self.master.winfo_screenheight()
        
        # Definir tamanho da janela
        self.window_width = int(self.screen_width)
        self.window_height = int(self.screen_height)
        self.master.geometry(f"{self.window_width}x{self.window_height}")
        
        # Variáveis
        self.video_path = None
        self.video = None
        self.first_frame = None
        self.mode = 'default'
        self.vagas = []
        self.pontos = []
        
        # Frame principal
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=1)
        
        # Widget para exibição de vídeo
        self.video_label = tk.Label(self.main_frame)
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        
        self.miniatura = tk.Label(self.main_frame)
        self.miniatura.pack(side=tk.LEFT, fill=tk.BOTH)

        # Barra lateral para botões
        self.sidebar = tk.Frame(self.main_frame, width=200, height=200)
        self.sidebar.pack(side=tk.TOP, fill=tk.X)
        
        # Botões
        self.btn_marcar = tk.Button(self.sidebar, text="Marcar Vagas", command=self.getVagas)
        self.btn_marcar.pack(pady=10, padx=10)
        
        self.btn_analisar = tk.Button(self.sidebar, text="Analisar Vídeo", command=self.detectVagas)
        self.btn_analisar.pack(pady=10, padx=10)
        
        # Carregar vídeo
        self.carregar_video()
    
    def carregar_video(self):
        # Solicitar caminho do vídeo
        if not self.video_path:
            self.video_path = filedialog.askopenfilename(title="Selecione o vídeo")
    
        # Captura de vídeo
        self.video = cv2.VideoCapture(self.video_path)
        ret, self.first_frame = self.video.read()
        if not ret:
            print("Erro ao ler o video")
            return
        
        # Mostrar o primeiro frame
        frame = self.first_frame.copy()
        frame = self.drawVagas(frame, self.processaFrame(frame))
        self.mostrar_frame(frame)
    
    def mostrar_frame(self, frame):
        # Definir tamanho do frame para ser x% do tamanho da janela
        frame_height, frame_width = frame.shape[:2]
        scale_percent = 0.75  # x% da janela
        new_width = int(self.window_width * scale_percent)
        new_height = int(frame_height * (new_width / frame_width))
        frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def mostra_miniatura(self, frame):
        # Definir tamanho do frame para ser x% do tamanho da janela
        frame_height, frame_width = frame.shape[:2]
        scale_percent = 0.15  # x% da janela
        new_width = int(self.window_width * scale_percent)
        new_height = int(frame_height * (new_width / frame_width))
        
        # Redimensionar o frame
        frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Converter frame de BGR para RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.miniatura.imgtk = imgtk
        self.miniatura.configure(image=imgtk)
    
    def getVagas(self):
        self.mode = 'marcar'
        # Implementar a lógica de marcação de vagas
        frame = self.first_frame.copy()
        frame = self.drawVagas(frame, self.processaFrame(frame))
        cv2.imshow('image', frame)
        cv2.setMouseCallback('image', self.click_event)

        # Press 'q' to leave
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        # Atualiza o tkinter
        frame = self.first_frame.copy()
        frame = self.drawVagas(frame, self.processaFrame(frame))
        self.mostrar_frame(frame)
    
    def detectVagas(self):
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break

            frame = self.drawVagas(frame, self.processaFrame(frame))
            
            self.mostrar_frame(frame)
            self.master.update_idletasks()
        
        self.video.release()
        self.carregar_video()

    def processaFrame(self, frame):
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar o algoritmo de Canny
        imagem_suavizada = cv2.GaussianBlur(grayFrame, (5, 5), 0)
        frame_processado = cv2.Canny(imagem_suavizada, 100, 200)

        self.mostra_miniatura(frame_processado)
        return frame_processado

    def drawVagas(self, frame, frame_processado):
        for vaga in self.vagas:
            recorte = self.recortar_Imagem(frame_processado, vaga.coords)
            whitePixels = cv2.countNonZero(recorte)
            relativeWhitePixels = whitePixels / self.calcArea(vaga.coords)
            # cv2.putText(frame, str(round(relativeWhitePixels, 2)), (vaga.coords[0][0], vaga.coords[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            
            recorte2 = self.recortar_Imagem(frame, vaga.coords)
            current_hist = calculate_histograms(recorte2)
            similarity = compare_histograms(vaga.initial_hist, current_hist)

            if similarity >= 0.6:
                vaga.ocupada = vaga.ocupada
            else:
                if relativeWhitePixels >= 0.15:
                    vaga.ocupada = True
                else:
                    vaga.ocupada = False
            
            vaga.register_occupation()


            ocupada = sum(1 for v in vaga.occupation_history if v) > len(vaga.occupation_history) / 2
            if ocupada:
                cv2.polylines(frame, [np.array(vaga.coords).reshape((-1, 1, 2))], isClosed=True, color=(0, 0, 255), thickness=2)
            else:
                cv2.polylines(frame, [np.array(vaga.coords).reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
        
        vagasOcupadas = sum(1 for vaga in self.vagas if vaga.ocupada)
        cv2.putText(frame, f"{vagasOcupadas}/{len(self.vagas)} Vagas", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    
        return frame
    
    def calcArea(self, pontos):
        x1, y1 = pontos[0]
        x2, y2 = pontos[1]
        x3, y3 = pontos[2]
        x4, y4 = pontos[3]
        
        area = 0.5 * abs(
            x1*y2 + x2*y3 + x3*y4 + x4*y1 -
            (y1*x2 + y2*x3 + y3*x4 + y4*x1)
        )
        
        return area

    def click_event(self, event, x, y, flags, params):
        frame = self.first_frame.copy()
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = self.drawVagas(frame, self.processaFrame(frame))
            self.pontos.append((x, y))
            
            if len(self.pontos) == 4:
                
                vaga = Vaga(self.pontos, self.question_popup(), calculate_histograms(self.recortar_Imagem(self.first_frame, self.pontos)))
                self.vagas.append(vaga)
                self.pontos = []
                frame = self.first_frame.copy()
                frame = self.drawVagas(frame, self.processaFrame(frame))
            cv2.imshow('image', frame)

    def question_popup(self):
        root = tk.Tk()
        root.withdraw()  # Oculta a janela principal

        result = messagebox.askokcancel("Vaga está ocupada?", "ok = sim ; cancelar = não")

        root.destroy()  # Fecha a janela principal

        return result

    def recortar_Imagem(self, imagem, coords):
        pts1 = np.float32(coords)
        largura = max(np.linalg.norm(pts1[0] - pts1[1]), np.linalg.norm(pts1[2] - pts1[3]))
        altura = max(np.linalg.norm(pts1[1] - pts1[2]), np.linalg.norm(pts1[3] - pts1[0]))

        pts2 = np.float32([[0, 0], [largura, 0], [largura, altura], [0, altura]])

        M = cv2.getPerspectiveTransform(pts1, pts2)

        recorte = cv2.warpPerspective(imagem, M, (int(largura), int(altura)))

        return recorte

root = tk.Tk()
app = VideoApp(root)
root.mainloop()
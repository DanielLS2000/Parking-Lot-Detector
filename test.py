import cv2
import numpy as np

# Função para garantir que o frame seja do tipo uint8
def garantir_uint8(frame):
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame

# Abrir o vídeo
video_path = 'parkin_timelapse.mp4'
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    raise ValueError("Não foi possível abrir o vídeo. Verifique o caminho do arquivo.")

# Ler o primeiro frame
ret, frame = video.read()

if not ret:
    raise ValueError("Não foi possível ler o frame do vídeo.")

# Garantir que o frame seja do tipo uint8
frame = garantir_uint8(frame)

# Converter para escala de cinza
imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Exibir a imagem em escala de cinza
cv2.imshow('Imagem em Escala de Cinza', imagem_cinza)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Fechar o vídeo
video.release()

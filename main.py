import cv2
import numpy as np


def calcArea(pontos):
    x1, y1 = pontos[0]
    x2, y2 = pontos[1]
    x3, y3 = pontos[2]
    x4, y4 = pontos[3]
    
    area = 0.5 * abs(
        x1*y2 + x2*y3 + x3*y4 + x4*y1 -
        (y1*x2 + y2*x3 + y3*x4 + y4*x1)
    )
    
    return area

def click_event(event, x, y, flags, params):
    global pontos, vagas, img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        pontos.append((x, y))
        
        if len(pontos) == 4:
            vagas.append(pontos)
            cv2.polylines(img, [np.array(pontos)], isClosed=True, color=(0, 255, 0), thickness=2)
            pontos = []
        
        cv2.imshow('image', img)

def recortar_Imagem(imagem, coords):
    pts1 = np.float32(coords)
    largura = max(np.linalg.norm(pts1[0] - pts1[1]), np.linalg.norm(pts1[2] - pts1[3]))
    altura = max(np.linalg.norm(pts1[1] - pts1[2]), np.linalg.norm(pts1[3] - pts1[0]))

    pts2 = np.float32([[0, 0], [largura, 0], [largura, altura], [0, altura]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    recorte = cv2.warpPerspective(imagem, M, (int(largura), int(altura)))

    return recorte

def remove_ruido(thresh):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    min_size = 80  # Tamanho mínimo de componente para manter
    cleaned_image = np.zeros(thresh.shape, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_image[labels == i] = 255
    return cleaned_image

# Registrando as Vagas
img = cv2.imread('photo.png')
vagas = []
pontos = []

cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)

# Press 'q' to leave
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()



# Detectando Vagas
video = cv2.VideoCapture("video.mp4")


output_video_path = 'output_video2.avi'
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=True)

while True:
    check, frame = video.read()
    
    if not check:
        break

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(grayFrame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 9)
    imgBlur = cv2.medianBlur(thresh, 5)
    kernel = np.ones((3,3), np.int8)
    imgD = cv2.morphologyEx(imgBlur, cv2.MORPH_OPEN, kernel)
    #imgD = remove_ruido(thresh)

    for vaga in vagas:
        frame2 = recortar_Imagem(imgD, vaga)
        whitePixels = cv2.countNonZero(frame2)
        relativeWhitePixels = whitePixels / calcArea(vaga)
        cv2.putText(frame, str(round(relativeWhitePixels, 2)), (vaga[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        
        if relativeWhitePixels >= 0.2:
            cv2.polylines(frame, [np.array(vaga)], isClosed=True, color=(0, 0, 255), thickness=2)
        else:
            cv2.polylines(frame, [np.array(vaga)], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow('Estacionamento', frame)
    cv2.imshow("Thresh", imgD)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
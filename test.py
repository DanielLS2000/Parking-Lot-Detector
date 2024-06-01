import cv2

# Nome do arquivo de vídeo de entrada
input_video_path = 'parkin_timelapse.mp4'
# Nome do arquivo de vídeo de saída
output_video_path = 'output_video.avi'

# Abrir o vídeo de entrada
cap = cv2.VideoCapture(input_video_path)

# Verificar se o vídeo foi aberto com sucesso
if not cap.isOpened():
    print("Erro ao abrir o vídeo de entrada.")
    exit()

# Obter a largura e altura dos frames do vídeo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Definir o codec e criar o objeto VideoWriter
# O codec 'XVID' cria arquivos .avi. Você pode mudar para 'mp4v' para arquivos .mp4
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

# Processar e salvar cada frame do vídeo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Aplicar algum processamento na imagem
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Escrever o frame processado no arquivo de vídeo de saída
    out.write(gray_frame)

    # Exibir o frame (opcional)
    cv2.imshow('Frame', gray_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os objetos de captura e escrita
cap.release()
out.release()
cv2.destroyAllWindows()

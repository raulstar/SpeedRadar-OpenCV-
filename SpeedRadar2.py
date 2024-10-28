import cv2  # Importa a biblioteca OpenCV para manipulação de vídeos e imagens
from tracker2 import *  # Importa todos os componentes do arquivo tracker2
import numpy as np  # Importa a biblioteca NumPy para manipulação de arrays

end = 0  # Inicializa a variável 'end' com valor 0

global tempo_inicio

# Cria o objeto do rastreador
tracker = EuclideanDistTracker()

# cap = cv2.VideoCapture("Resources/traffic3.mp4")  # Comentado: Abre um vídeo específico no caminho "Resources/traffic3.mp4"
cap = cv2.VideoCapture(1)  # fonte do video"

f = 30  # Define a variável 'f' com valor 25 (frames por segundo)
w = int(1000 / (f - 1))                                                      # Calcula o tempo entre quadros

# Detecção de Objetos
object_detector = cv2.createBackgroundSubtractorMOG2(history=None, varThreshold=None)        # Cria o subtrator de fundo MOG2

# KERNELS   
kernalOp = np.ones((3, 3), np.uint8)                                        # Cria uma matriz de uns 3x3 para operações morfológicas de abertura
kernalOp2 = np.ones((5, 5), np.uint8)                                       # Cria uma matriz de uns 5x5 para operações morfológicas de abertura
kernalCl = np.ones((11, 11), np.uint8)                                       # Cria uma matriz de uns 11x11 para operações morfológicas de fechamento
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)               # Cria o subtrator de fundo MOG2 com detecção de sombras
kernal_e = np.ones((5, 5), np.uint8)                                        # Cria uma matriz de uns 5x5 para operações morfológicas de erosão

while True:                                                                 # Inicia um loop infinito
    ret, frame = cap.read()                                                 # Lê um frame do vídeo
    if not ret:                                                             # Se não conseguir ler o frame (fim do vídeo), sai do loop
        break
    frame = cv2.resize(frame, None, fx=1, fy=1)                             # Redimensiona o frame pela metade
    height, width, _ = frame.shape                                           # Obtém a altura e a largura do frame

    # print(height, width)                                                   # (Comentado) Imprime a altura e largura do frame
    # 540, 960                                                              # (Comentado) Exemplo de dimensões possíveis

    # Extract ROI                                                            # Extrai a Região de Interesse (ROI)
    roi = frame[50:440, 180:560]                                             # [50:540,200:960]

    # MASKING METHOD 1                                                       # Método de mascaramento 1
    mask = object_detector.apply(roi)                                       # Aplica o detector de objetos à ROI
    _, mask = cv2.threshold(
        mask, 250, 255, cv2.THRESH_BINARY
    )  # Aplica um limiar binário à máscara

    # DIFFERENT MASKING METHOD 2 -> This is used                            # Método de mascaramento 2 -> Este é utilizado
    fgmask = fgbg.apply(roi)                                                 # Aplica o subtrator de fundo à ROI
    ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY )        # Aplica um limiar binário à máscara
    mask1 = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)                # Realiza uma operação morfológica de abertura
    mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernalCl)              # Realiza uma operação morfológica de fechamento
    e_img = cv2.erode(mask2, kernal_e)                                      # Erosiona a imagem

    contours, _ = cv2.findContours(e_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Encontra contornos na imagem
    detections = []                                                          # Lista para armazenar as detecções

    for cnt in contours:                                                     # Para cada contorno encontrado
        area = cv2.contourArea(cnt)                                         # Calcula a área do contorno
        # THRESHOLD                                                          # Limite
        if area > 700:                                                      # Se a área for maior que 1000
            x, y, w, h = cv2.boundingRect(cnt)                               # Obtém o retângulo delimitador
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)      # Desenha um retângulo ao redor do objeto
            detections.append([x, y, w, h])                                 # Adiciona a detecção à lista

    # Object Tracking
    boxes_ids = tracker.update(detections)                                  # Atualiza o rastreador com as novas detecções
    
    for box_id in boxes_ids:                                                 # Para cada ID de caixa rastreada
        x, y, w, h, id = box_id                                             # Desempacota as coordenadas e o ID
        if (tracker.getsp(id) < tracker.limit()):                             # Se a velocidade do objeto for menor que o limite
            cv2.putText(roi, str(id) + " " + str(tracker.getsp(id)), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, )  # Adiciona o texto com o ID e velocidade (amarelo)
            cv2.rectangle( roi, (x, y), (x + w, y + h), (0, 255, 0), 3 )     # Desenha um retângulo verde ao redor do objeto
        else:
            cv2.putText(roi, str(id) + " " + str(tracker.getsp(id)), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, )  # Adiciona o texto com o ID e velocidade (vermelho)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 165, 255), 3 )    # Desenha um retângulo laranja ao redor do objeto

        s = tracker.getsp(id)                                                # Obtém a velocidade do objeto
        if (tracker.f[id] == 1 and s != 0):                                 # Se a flag do objeto for 1 e a velocidade não for 0
            tracker.capture(roi, x, y, h, w, s, id)                             # Captura a ROI do objeto

    # DRAW LINES
    cv2.line(roi, (0, inicio_max), (960, inicio_max), (0, 0, 255), 2)
    cv2.line(roi, (0, inicio_mim), (960, inicio_mim), (0, 0, 255), 2)

    cv2.line(roi, (0, final_max), (960, final_max), (0, 0, 255), 2)
    cv2.line(roi, (0, final_mim), (960, final_mim), (0, 255, 255), 2)

    # DISPLAY
    # cv2.imshow("Mask", mask2)                                             # (Comentado) Mostra a máscara 'mask2'
    # cv2.imshow("Erode", e_img)                                             # (Comentado) Mostra a imagem erodida 'e_img'
    cv2.imshow("ROI", roi)                                                   # Mostra a Região de Interesse 'ROI'
    key = cv2.waitKey(w - 10)                                                # Aguarda por uma tecla pressionada com atraso de 'w-10'
    if cv2.waitKey(1) == ord("q"):                                          # Se a tecla 'Esc' for pressionada
        tracker.end()                                                        # Termina o rastreamento
        end = 1                                                             # Define a variável 'end' como 1
        break

if end != 1:                                                                 # Se a variável 'end' não for igual a 1
    tracker.end()                                                               # Termina o rastreamento

cap.release()                                                                # Libera o vídeo capturado
cv2.destroyAllWindows()                                                     # Fecha todas as janelas do OpenCV

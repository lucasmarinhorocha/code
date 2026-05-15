from ultralytics import YOLO
import cv2
import serial
import time
from collections import defaultdict

# ─────────────────────────────────────────
# CONFIGURAÇÃO DO ARDUINO
# ─────────────────────────────────────────
print("Tentando conectar ao Arduino...")
try:
    arduino = serial.Serial('COM6', 9600)
    time.sleep(2)
    print("✅ Conectado ao Arduino")
except Exception as e:
    print(f"⚠️ Erro na conexão Serial: {e}")
    arduino = None

# ─────────────────────────────────────────
# MODELO E CÂMERA (COM TRY-EXCEPT)
# ─────────────────────────────────────────
print("Carregando modelo YOLO...")
modelo = YOLO("Modelo/best.pt")

print("📸 Inicializando câmera...")
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Não foi possível abrir o índice 0 da câmera.")
    print("✅ Câmera conectada com sucesso!")
except Exception as e:
    print(f"❌ Erro ao ligar a câmera: {e}")
    exit()

# ─────────────────────────────────────────
# LÓGICA DE ESTABILIZAÇÃO (DEBOUNCE)
# ─────────────────────────────────────────
DEBOUNCE_FRAMES = 20  
frames_ausente = {'Garrafa': 0, 'Lata': 0, 'Papel': 0}
estado_enviado = {'Garrafa': False, 'Lata': False, 'Papel': False}

def processar_deteccoes(contagem_agora):
    global frames_ausente, estado_enviado

    config_classes = [
        ('Garrafa', b'1'),
        ('Lata',    b'2'),
        ('Papel',   b'3'),
    ]

    # Verifica se já existe algum objeto "em processo"
    algum_ativo = any(estado_enviado.values())
    
    # LÓGICA DE PRIORIDADE: Só processamos um novo se não houver nenhum ativo
    for classe, sinal_presente in config_classes:
        tem_agora = contagem_agora[classe] > 0
        
        if tem_agora:
            frames_ausente[classe] = 0
            
            # Só envia o comando se:
            # 1. Esta classe ainda não foi enviada
            # 2. NÃO existe outra classe ativa no momento (evita o conflito)
            if not estado_enviado[classe] and not algum_ativo:
                if arduino:
                    arduino.write(sinal_presente)
                print(f"✅ [CONTROLE] {classe} detectado. Enviando sinal {sinal_presente.decode()}")
                estado_enviado[classe] = True
                algum_ativo = True # Bloqueia outras classes neste frame
        else:
            frames_ausente[classe] += 1

    # Lógica para voltar ao ZERO (Só se o que estava ativo sumiu)
    for classe, _ in config_classes:
        if estado_enviado[classe] and frames_ausente[classe] >= DEBOUNCE_FRAMES:
            if arduino:
                arduino.write(b'0')
            print(f"🔄 [RESETE] {classe} saiu. Voltando ao zero.")
            estado_enviado[classe] = False

# ─────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────
print("🚀 Iniciando loop de detecção...")
while True:
    ret, img = cap.read()
    if not ret: 
        print("⚠️ Falha ao capturar frame.")
        break

    resultados = modelo.track(img, persist=True, verbose=False, conf=0.5)[0]
    contagem_agora = {'Garrafa': 0, 'Lata': 0, 'Papel': 0}

    if resultados.boxes.id is not None:
        classes = resultados.boxes.cls.cpu().tolist()
        for cls in classes:
            nome_classe = modelo.names[int(cls)]
            if nome_classe in contagem_agora:
                contagem_agora[nome_classe] += 1

    processar_deteccoes(contagem_agora)

    # Interface Visual
    img_anotada = resultados.plot()
    for i, (classe, _) in enumerate([('Garrafa','1'), ('Lata','2'), ('Papel','3')]):
        cor = (0, 255, 0) if estado_enviado[classe] else (0, 0, 255)
        cv2.putText(img_anotada, f"{classe}: {frames_ausente[classe]}/{DEBOUNCE_FRAMES}", 
                    (10, 40 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

    cv2.imshow("Sistema de Triagem", img_anotada)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
if arduino: arduino.close()
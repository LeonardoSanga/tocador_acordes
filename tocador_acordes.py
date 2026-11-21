"""
TOCADOR DE ACORDES POR GESTOS
Visao computacional com OpenCV + MediaPipe + Pygame

Como tocar:
  - MAO ESQUERDA (lado esquerdo da tela): quantos dedos levantados = GRAU do acorde
        1 dedo = I, 2 dedos = II, 3 dedos = III, 4 dedos = IV, 5 dedos = V
  - MAO DIREITA (lado direito da tela): forma da mao = QUALIDADE do acorde
        "L" (polegar + indicador) = MAIOR
        mao aberta (5 dedos)      = MENOR
        punho fechado             = SILENCIO

Teclas:  q = sair | t = trocar timbre (pad, violao, piano, violino)
         + / - = mudar o tom | c = proxima cola (Creep, Let It Be, Stand By Me, Zombie)

Uso:
  python tocador_acordes.py            -> usa a webcam
  python tocador_acordes.py video.mp4  -> usa um arquivo de video
"""

import math
import os
import sys
import time
import urllib.request

import cv2
import numpy as np
import pygame
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# 1) CONFIGURACOES
ARQUIVO_MODELO = 'hand_landmarker.task'
URL_MODELO = ('https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
              'hand_landmarker/float16/latest/hand_landmarker.task')

TOM_INICIAL = 'G'          # Creep e em Sol (G)
TIMBRE_INICIAL = 'pad'     # 'pad', 'violao', 'piano' ou 'violino'
BPM = 92                   # andamento de Creep (usado no modo violao)
FRAMES_ESTAVEIS = 4        # quantos frames seguidos o gesto precisa durar

TAXA_AMOSTRAGEM = 44100    # amostras de audio por segundo


# 2) TEORIA MUSICAL
NOTAS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
ESCALA_MAIOR = [0, 2, 4, 5, 7, 9, 11]   # distancia (em semitons) de cada grau ate a tonica
ROMANOS = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']


def montar_acorde(tom, grau, qualidade):
    """Recebe o tom (ex.: 'G'), o grau (1 a 5) e a qualidade ('maior'/'menor').
    Devolve o nome do acorde (ex.: 'Cm') e o indice da nota fundamental (0 a 11)."""
    tonica = NOTAS.index(tom)
    fundamental = (tonica + ESCALA_MAIOR[grau - 1]) % 12
    nome = NOTAS[fundamental] + ('m' if qualidade == 'menor' else '')
    return nome, fundamental


def numeral_romano(grau, qualidade):
    r = ROMANOS[grau - 1]
    return r if qualidade == 'maior' else r.lower()


def frequencia(nota_midi):
    """Converte numero MIDI em Hz. A4 (MIDI 69) = 440 Hz; cada semitom multiplica por 2^(1/12)."""
    return 440.0 * 2 ** ((nota_midi - 69) / 12)


def notas_do_acorde(fundamental, qualidade):
    """Monta o acorde como num violao (formato de pestana 'de E'): 6 notas."""
    baixo = 40 + (fundamental - 4) % 12             # nota mais grave entre E2 (40) e D#3 (51)
    terca = 4 if qualidade == 'maior' else 3          # a TERCA e o que diferencia maior de menor
    intervalos = [0, 7, 12, 12 + terca, 19, 24]       # fundamental, quinta, oitava, terca, quinta, oitava
    return [baixo + i for i in intervalos]


# 3) SINTESE DE SOM (gerar audio com numpy, sem arquivos .wav)
def gerar_pad(midis, duracao=4.0):
    """Som de sintetizador continuo ('Warm Synth'). Feito para tocar em loop sem estalos:
    cada frequencia e arredondada para caber um numero inteiro de ciclos em 'duracao'."""
    n = int(TAXA_AMOSTRAGEM * duracao)
    t = np.arange(n) / TAXA_AMOSTRAGEM
    esquerda = np.zeros(n)
    direita = np.zeros(n)
    for m in midis:
        for desafino, lado in [(-0.6, 'e'), (0.6, 'd')]:      # duas vozes levemente desafinadas
            f = frequencia(m) + desafino
            onda = np.zeros(n)
            for h in range(1, 7):                              # harmonicos (deixam o som "quente")
                fh = round(f * h * duracao) / duracao          # numero inteiro de ciclos -> loop perfeito
                onda += np.sin(2 * np.pi * fh * t) / h ** 1.6
            if lado == 'e':
                esquerda += onda
            else:
                direita += onda
    tremolo = 1 + 0.08 * np.sin(2 * np.pi * (2 / duracao) * t)  # leve "respiracao" do som
    estereo = np.stack([esquerda * tremolo, direita * tremolo], axis=1)
    return estereo / np.max(np.abs(estereo)) * 0.5


def corda_karplus_strong(f, n):
    """Simula uma corda dedilhada: ruido aleatorio que passa repetidamente
    por um filtro de media -> vira um tom que decai como uma corda real."""
    periodo = max(2, int(round(TAXA_AMOSTRAGEM / f - 0.5)))
    y = np.zeros(n)
    y[:periodo + 1] = np.random.uniform(-1, 1, periodo + 1)
    amortecimento = 0.996
    inicio = periodo + 1
    while inicio < n:
        fim = min(inicio + periodo, n)
        y[inicio:fim] = amortecimento * 0.5 * (y[inicio - periodo:fim - periodo] +
                                               y[inicio - periodo - 1:fim - periodo - 1])
        inicio = fim
    return y


def gerar_violao(midis, duracao=2.5):
    """Acorde 'batido': as 6 cordas soam com um pequeno atraso entre si."""
    n = int(TAXA_AMOSTRAGEM * duracao)
    som = np.zeros(n)
    atraso = int(0.018 * TAXA_AMOSTRAGEM)               # 18 ms entre uma corda e a proxima
    for i, m in enumerate(midis):
        corda = corda_karplus_strong(frequencia(m), n - i * atraso)
        som[i * atraso:] += corda * (1.0 if i == 0 else 0.8)
    fade = np.ones(n)
    fade[-2000:] = np.linspace(1, 0, 2000)               # evita estalo no final
    som *= fade
    som = som / np.max(np.abs(som)) * 0.6
    return np.stack([som, som], axis=1)


def gerar_piano(midis, duracao=4.0):
    """Piano: cada nota e uma soma de harmonicos que DECAEM (o martelo bate e solta).
    Os harmonicos agudos somem mais rapido, e cada um fica um pouco 'desafinado'
    para cima (inarmonicidade), como numa corda de aco esticada."""
    n = int(TAXA_AMOSTRAGEM * duracao)
    t = np.arange(n) / TAXA_AMOSTRAGEM
    som = np.zeros(n)
    for m in midis[1:]:                                   # 5 notas: sem a mais grave fica mais limpo
        f = frequencia(m)
        for k in range(1, 11):
            fk = k * f * math.sqrt(1 + 0.0004 * k * k)    # inarmonicidade da corda de piano
            if fk > TAXA_AMOSTRAGEM / 2:
                break
            decaimento = 0.8 + 0.45 * k + f / 600         # agudos e harmonicos altos decaem antes
            som += np.sin(2 * np.pi * fk * t) * np.exp(-decaimento * t) / k ** 1.3
    ataque = np.minimum(1, t / 0.004)                     # 4 ms de ataque (o martelo)
    martelo = np.random.uniform(-1, 1, n) * np.exp(-t / 0.01) * 0.15
    som = (som * ataque) + martelo
    fade = np.ones(n)
    fade[-3000:] = np.linspace(1, 0, 3000)
    som = som * fade / np.max(np.abs(som)) * 0.6
    return np.stack([som, som], axis=1)


def gerar_violino(midis, duracao=4.0):
    """Violinos (naipe de cordas): onda 'dente de serra' (todos os harmonicos, 1/k)
    com VIBRATO, o leve sobe-e-desce de afinacao que o violinista faz com o dedo.
    Como no pad, tudo tem um numero inteiro de ciclos para tocar em loop sem estalo."""
    n = int(TAXA_AMOSTRAGEM * duracao)
    t = np.arange(n) / TAXA_AMOSTRAGEM
    vib = 5.0                                              # vibrato de 5 Hz (20 ciclos em 4 s)
    esquerda = np.zeros(n)
    direita = np.zeros(n)
    for i, m in enumerate(midis[2:]):                      # 4 notas mais agudas: regiao do violino
        for voz in range(2):                               # 2 violinistas por nota
            f = round(frequencia(m) * (1 + (voz - 0.5) * 0.004) * duracao) / duracao
            fase_vib = 1.3 * i + 2.1 * voz
            profundidade = 0.004 * f                       # vibrato de +-0,4%
            onda = np.zeros(n)
            for k in range(1, 13):
                fk = k * f
                if fk > 9000:
                    break
                corpo = 1 + 1.5 * math.exp(-((fk - 2800) / 1200) ** 2)  # ressonancia do corpo
                fase = 2 * np.pi * fk * t + (k * profundidade / vib) * np.sin(2 * np.pi * vib * t + fase_vib)
                onda += np.sin(fase) * corpo / k
            if voz == 0:
                esquerda += onda
            else:
                direita += onda
    estereo = np.stack([esquerda, direita], axis=1)
    return estereo / np.max(np.abs(estereo)) * 0.45


# Como cada timbre se comporta:
#   continuo=True  -> toca em loop enquanto o gesto for mantido
#   continuo=False -> e "batido": toca de novo a cada 'tempos' batidas do BPM
TIMBRES = {
    'pad':     {'gerar': gerar_pad,     'continuo': True},
    'violao':  {'gerar': gerar_violao,  'continuo': False, 'tempos': 1},
    'piano':   {'gerar': gerar_piano,   'continuo': False, 'tempos': 1},
    'violino': {'gerar': gerar_violino, 'continuo': True},
}
ORDEM_TIMBRES = list(TIMBRES)


def para_pygame(audio):
    """Converte o array float (-1..1) em um objeto Sound do pygame (16 bits, estereo)."""
    inteiro = np.ascontiguousarray((audio * 32767).astype(np.int16))
    return pygame.sndarray.make_sound(inteiro)


class BancoDeSons:
    """Guarda os sons ja gerados (cache) para nao recalcular a cada frame."""

    def __init__(self):
        self.cache = {}

    def obter(self, fundamental, qualidade, timbre):
        chave = (fundamental, qualidade, timbre)
        if chave not in self.cache:
            midis = notas_do_acorde(fundamental, qualidade)
            audio = TIMBRES[timbre]['gerar'](midis)
            self.cache[chave] = para_pygame(audio)
        return self.cache[chave]


# 4) LEITURA DA MAO (os 21 pontos do MediaPipe)
#   0 = pulso | 1-4 = polegar | 5-8 = indicador | 9-12 = medio
#   13-16 = anelar | 17-20 = minimo   (o ultimo numero de cada dedo e a PONTA)
CONEXOES_MAO = [(0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def distancia(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dedos_levantados(pontos):
    """Devolve [polegar, indicador, medio, anelar, minimo] com True/False."""
    pulso = pontos[0]
    palma = distancia(pontos[0], pontos[9])           # tamanho da mao (serve de regua)

    # Polegar: esticado quando a ponta (4) fica longe da base do indicador (5)
    # e mais longe do dedo minimo (17) do que a propria junta do polegar (3).
    polegar = (distancia(pontos[4], pontos[5]) > 0.55 * palma and
               distancia(pontos[4], pontos[17]) > distancia(pontos[3], pontos[17]))

    # Outros dedos: esticado quando a ponta fica mais longe do pulso que a junta do meio.
    resultado = [polegar]
    for ponta, junta in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        resultado.append(distancia(pontos[ponta], pulso) > distancia(pontos[junta], pulso) * 1.1)
    return resultado


def ler_grau(dedos):
    """Mao do GRAU: so importa QUANTOS dedos estao levantados (1 a 5)."""
    total = sum(dedos)
    return total if 1 <= total <= 5 else None


def ler_qualidade(dedos):
    """Mao da QUALIDADE: reconhece tres formas."""
    polegar, indicador, medio, anelar, minimo = dedos
    if polegar and indicador and not (medio or anelar or minimo):
        return 'maior'      # "L"
    if sum(dedos) >= 4 and medio and anelar:
        return 'menor'      # mao aberta
    if sum(dedos) == 0:
        return 'silencio'   # punho fechado
    return None             # gesto desconhecido


# 5) DESENHO NA TELA
def desenhar_mao(imagem, pontos, cor):
    for a, b in CONEXOES_MAO:
        cv2.line(imagem, pontos[a], pontos[b], cor, 2)
    for i, p in enumerate(pontos):
        raio = 6 if i in (4, 8, 12, 16, 20) else 4
        cv2.circle(imagem, p, raio, (255, 255, 255), -1)
        cv2.circle(imagem, p, raio, cor, 1)


def caixa_da_mao(pontos, margem=20):
    xs = [p[0] for p in pontos]
    ys = [p[1] for p in pontos]
    return min(xs) - margem, min(ys) - margem, max(xs) + margem, max(ys) + margem


def texto(imagem, msg, pos, escala=0.6, cor=(255, 255, 255), espessura=1):
    cv2.putText(imagem, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), espessura + 3, cv2.LINE_AA)
    cv2.putText(imagem, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, espessura, cv2.LINE_AA)


# Colas (tecla c): cada musica tem o tom em que todos os acordes cabem nos graus I a V.
# A cola so mostra o tom indicado: o usuario muda o tom manualmente com + / -.
COLAS = [
    {'nome': 'CREEP', 'tom': 'G', 'linhas': [
        'G  = 1 dedo  + L         (2 compassos)',
        'B  = 3 dedos + L         (2 compassos)',
        'C  = 4 dedos + L         (2 compassos)',
        'Cm = 4 dedos + mao aberta (2 compassos)']},
    {'nome': 'LET IT BE', 'tom': 'F', 'linhas': [
        'C  = 5 dedos + L',
        'G  = 2 dedos + L',
        'Am = 3 dedos + mao aberta',
        'F  = 1 dedo  + L',
        'Verso: C G Am F | C G F C']},
    {'nome': 'STAND BY ME', 'tom': 'D', 'linhas': [
        'A   = 5 dedos + L',
        'F#m = 3 dedos + mao aberta',
        'D   = 1 dedo  + L',
        'E   = 2 dedos + L',
        'Ciclo: A F#m D E A']},
    {'nome': 'ZOMBIE', 'tom': 'C', 'linhas': [
        'Em = 3 dedos + mao aberta',
        'C  = 1 dedo  + L',
        'G  = 5 dedos + L',
        'D  = 2 dedos + L',
        'Ciclo: Em C G D']},
]


def desenhar_painel(imagem, acorde, romano, tom, timbre, cola):
    h, w = imagem.shape[:2]
    faixa = imagem.copy()
    cv2.rectangle(faixa, (0, 0), (w, 40), (30, 30, 30), -1)
    cv2.addWeighted(faixa, 0.6, imagem, 0.4, 0, imagem)
    texto(imagem, f'Tom: {tom}   Timbre: {timbre}   [q] sair  [t] timbre  [+/-] tom  [c] cola',
          (10, 27), 0.55)

    if acorde:
        tamanho = cv2.getTextSize(acorde, cv2.FONT_HERSHEY_DUPLEX, 3, 5)[0]
        x = (w - tamanho[0]) // 2
        cv2.putText(imagem, acorde, (x, 130), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 12, cv2.LINE_AA)
        cv2.putText(imagem, acorde, (x, 130), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 200, 255), 5, cv2.LINE_AA)
        texto(imagem, f'({romano})', (x + tamanho[0] + 10, 130), 1.0, (0, 200, 255), 2)

    if cola is not None:
        linhas = [f"{cola['nome']} (tom {cola['tom']})   [c] proxima cola"] + cola['linhas']
        for i, linha in enumerate(linhas):
            texto(imagem, linha, (10, h - 20 - 26 * (len(linhas) - 1 - i)), 0.6, (180, 255, 180))


# 6) PROGRAMA PRINCIPAL
def carregar_detector():
    # Assim como o haarcascade_*.xml nas aulas, o MediaPipe precisa de um arquivo de modelo.
    if not os.path.exists(ARQUIVO_MODELO):
        print('Baixando o modelo de maos (so na primeira vez)...')
        try:
            urllib.request.urlretrieve(URL_MODELO, ARQUIVO_MODELO)
        except Exception as erro:
            print('Nao consegui baixar automaticamente:', erro)
            print('Baixe manualmente pelo navegador e salve nesta pasta como', ARQUIVO_MODELO)
            print(URL_MODELO)
            sys.exit(1)
    opcoes = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ARQUIVO_MODELO),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    return vision.HandLandmarker.create_from_options(opcoes)


def main():
    pygame.mixer.init(frequency=TAXA_AMOSTRAGEM, size=-16, channels=2, buffer=512)
    pygame.mixer.set_num_channels(16)
    sons = BancoDeSons()

    detector = carregar_detector()
    fonte = sys.argv[1] if len(sys.argv) > 1 else 0
    video = cv2.VideoCapture(fonte)

    tom = TOM_INICIAL
    timbre = TIMBRE_INICIAL
    indice_cola = 0            # comeca mostrando a cola de Creep

    acorde_atual = None        # o que esta tocando agora: (nome, fundamental, qualidade, romano)
    candidato = None           # o que a camera esta vendo neste momento
    contador = 0               # ha quantos frames o candidato se repete
    som_atual = None
    proxima_batida = 0.0       # (modo violao) quando bater o acorde de novo
    inicio = time.perf_counter()
    ultimo_tempo = -1

    while True:
        conectado, frame = video.read()
        if not conectado:
            break

        frame = cv2.flip(frame, 1)                       # espelha: a tela vira um espelho
        h, w = frame.shape[:2]

        # --- Deteccao das maos ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # OpenCV usa BGR, MediaPipe usa RGB
        imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        tempo_ms = max(int((time.perf_counter() - inicio) * 1000), ultimo_tempo + 1)
        ultimo_tempo = tempo_ms
        resultado = detector.detect_for_video(imagem_mp, tempo_ms)

        # Converte os 21 pontos (0 a 1) de cada mao para pixels
        maos = []
        for marcos in resultado.hand_landmarks:
            maos.append([(int(p.x * w), int(p.y * h)) for p in marcos])
        maos.sort(key=lambda pontos: pontos[0][0])       # ordena da esquerda para a direita

        # --- Interpretacao dos gestos ---
        grau = qualidade = None
        if len(maos) == 2:
            mao_grau, mao_qualidade = maos[0], maos[1]
            dedos_g = dedos_levantados(mao_grau)
            dedos_q = dedos_levantados(mao_qualidade)
            grau = ler_grau(dedos_g)
            qualidade = ler_qualidade(dedos_q)
            rotulos = [(mao_grau, (255, 120, 0), f'GRAU: {sum(dedos_g)} dedo(s)'),
                       (mao_qualidade, (0, 200, 0), f'QUALIDADE: {qualidade or "?"}')]
        else:
            rotulos = [(p, (0, 0, 255), 'mostre as duas maos') for p in maos]

        for pontos, cor, rotulo in rotulos:
            x1, y1, x2, y2 = caixa_da_mao(pontos)
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)   # igual ao detector de rostos!
            desenhar_mao(frame, pontos, cor)
            texto(frame, rotulo, (x1, max(y1 - 10, 60)), 0.6, cor, 2)

        # --- Gesto -> acorde ---
        if len(maos) < 2 or qualidade == 'silencio':
            detectado = 'silencio'
        elif grau and qualidade in ('maior', 'menor'):
            nome, fundamental = montar_acorde(tom, grau, qualidade)
            detectado = (nome, fundamental, qualidade, numeral_romano(grau, qualidade))
        else:
            detectado = None   # gesto incompleto: mantem o que esta tocando

        # --- Filtro de estabilidade ---
        if detectado == candidato:
            contador += 1
        else:
            candidato = detectado
            contador = 1

        agora = time.perf_counter()
        if contador == FRAMES_ESTAVEIS and candidato is not None:
            novo = None if candidato == 'silencio' else candidato
            if novo != acorde_atual:
                if som_atual:
                    som_atual.fadeout(250)
                    som_atual = None
                acorde_atual = novo
                if acorde_atual:
                    _, fundamental, qual, _ = acorde_atual
                    som_atual = sons.obter(fundamental, qual, timbre)
                    config = TIMBRES[timbre]
                    if config['continuo']:
                        ataque = 250 if timbre == 'violino' else 120   # o arco "entra" devagar
                        som_atual.play(loops=-1, fade_ms=ataque)
                    else:
                        som_atual.play()
                        proxima_batida = agora + config['tempos'] * 60 / BPM

        # Timbres batidos (violao, piano): enquanto o gesto for mantido, toca de novo no ritmo
        config = TIMBRES[timbre]
        if not config['continuo'] and acorde_atual and agora >= proxima_batida:
            som_atual.fadeout(60)
            som_atual.play()
            proxima_batida = agora + config['tempos'] * 60 / BPM

        nome = acorde_atual[0] if acorde_atual else None
        romano = acorde_atual[3] if acorde_atual else ''
        cola = COLAS[indice_cola] if indice_cola < len(COLAS) else None
        desenhar_painel(frame, nome, romano, tom, timbre, cola)
        cv2.imshow('Tocador de Acordes', frame)

        # --- Teclado ---
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        mudou = False
        if tecla == ord('t'):
            timbre = ORDEM_TIMBRES[(ORDEM_TIMBRES.index(timbre) + 1) % len(ORDEM_TIMBRES)]
            mudou = True
        elif tecla in (ord('+'), ord('=')):
            tom = NOTAS[(NOTAS.index(tom) + 1) % 12]
            mudou = True
        elif tecla in (ord('-'), ord('_')):
            tom = NOTAS[(NOTAS.index(tom) - 1) % 12]
            mudou = True
        elif tecla == ord('c'):
            # Creep -> Let It Be -> Stand By Me -> Zombie -> sem cola -> Creep ...
            # (a cola so mostra o tom indicado; quem muda o tom e o usuario, com + / -)
            indice_cola = (indice_cola + 1) % (len(COLAS) + 1)
        if mudou:                     # recomeca o acorde com o novo tom/timbre
            pygame.mixer.fadeout(150)
            som_atual = None
            acorde_atual = None
            candidato = None

    video.release()
    cv2.destroyAllWindows()
    detector.close()
    pygame.mixer.quit()


if __name__ == '__main__':
    main()
# 🎸 Tocador de Acordes por Gestos

> Toque música com as mãos, sem encostar em nada: a webcam lê os seus gestos e o programa toca acordes em tempo real.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-visão%20computacional-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Landmarker-0097A7?logo=google&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-síntese%20de%20áudio-013243?logo=numpy&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-reprodução%20de%20som-2E7D32)

Projeto de visão computacional desenvolvido para a faculdade. Com as duas mãos na frente da câmera, você forma acordes: **a mão esquerda escolhe o acorde da escala e a mão direita escolhe se ele é maior ou menor.**

<!-- Sugestão: grave a tela tocando e coloque um GIF aqui -->
<!-- ![Demonstração](docs/demo.gif) -->

---

## ✨ Funcionalidades

- 🖐️ **Detecção das duas mãos em tempo real**, com 21 pontos por mão (MediaPipe).
- 🎵 **Gestos viram acordes**: 5 graus da escala × maior/menor, em qualquer um dos 12 tons.
- 🔊 **Som 100% gerado por código**: nenhum arquivo de áudio, as ondas são calculadas com NumPy.
- 🎹 **4 timbres**: pad (sintetizador), violão, piano e violino.
- 🛡️ **Filtro de estabilidade**: o acorde só troca quando o gesto se mantém por alguns frames, sem "piscar".
- 📋 **Colas na tela** para *Creep*, *Let It Be*, *Stand By Me* e *Zombie*.
- 🎓 **Versões em etapas** para quem quer aprender a construir o programa do zero.

---

## 🧰 Tecnologias

| Tecnologia | Para que serve no projeto |
|---|---|
| [Python](https://www.python.org/) | Linguagem do projeto |
| [OpenCV](https://opencv.org/) | Captura da webcam, espelhamento, conversão de cores, desenho na tela e teclado (as mesmas funções das aulas: `VideoCapture`, `cvtColor`, `rectangle`, `putText`, `imshow`, `waitKey`) |
| [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) | Modelo de rede neural *Hand Landmarker*, que encontra 21 pontos em cada mão |
| [NumPy](https://numpy.org/) | Síntese do som: cálculo das ondas de cada acorde |
| [Pygame](https://www.pygame.org/) | Reprodução do áudio, com loop e *fade* |

---

## 🖐️ Como tocar

O vídeo aparece espelhado: sua mão esquerda fica do lado esquerdo da tela.

| Mão | Gesto | Significado |
|---|---|---|
| **Esquerda** | 1 a 5 dedos levantados (o polegar conta) | Grau do acorde: I, II, III, IV ou V |
| **Direita** | "L" (polegar + indicador) | Acorde **maior** |
| **Direita** | Mão aberta | Acorde **menor** |
| **Direita** | Punho fechado | **Silêncio** |

O som só sai com as duas mãos visíveis. No tom de G (Sol), que é o tom inicial, os gestos produzem:

| Dedos (esquerda) | Grau | Com "L" | Com mão aberta |
|:---:|:---:|:---:|:---:|
| 1 | I | G | Gm |
| 2 | II | A | Am |
| 3 | III | B | Bm |
| 4 | IV | C | Cm |
| 5 | V | D | Dm |

### Teclas

| Tecla | Ação |
|:---:|---|
| `q` | Sair |
| `t` | Trocar o timbre: pad → violão → piano → violino |
| `+` / `-` | Subir ou descer o tom em meio tom |
| `c` | Trocar a cola na tela: Creep → Let It Be → Stand By Me → Zombie → sem cola |

---

## 🚀 Como rodar

### Requisitos

- Python 3.10 ou mais novo (testado com Python 3.13 e MediaPipe 0.10.32)
- Uma webcam
- Caixa de som ou fone de ouvido

### Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/SEU-USUARIO/tocador-de-acordes.git
cd tocador-de-acordes

# 2. (Opcional, recomendado) Crie um ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt
```

### Execução

```bash
python tocador_acordes.py
```

Na primeira execução, o programa baixa sozinho o modelo `hand_landmarker.task` (alguns MB) para a pasta do projeto.

Também dá para usar um arquivo de vídeo no lugar da webcam:

```bash
python tocador_acordes.py meu_video.mp4
```

---

## 📄 Licença

Distribuído sob a licença MIT. Veja o arquivo [`LICENSE`](LICENSE).

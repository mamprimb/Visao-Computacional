import numpy as np
from PIL import Image

def aplicar_filtro_mediana(imagem, tamanho_kernel):
    img = np.array(imagem)
    altura, largura, canais = img.shape

    img_filtrada = np.zeros_like(img)

    deslocamento = tamanho_kernel // 2

    for i in range(deslocamento, altura - deslocamento):
        for j in range(deslocamento, largura - deslocamento):
            for c in range(canais):
                regiao = img[i - deslocamento:i + deslocamento + 1, j - deslocamento:j + deslocamento + 1, c]
                
                img_filtrada[i, j, c] = np.median(regiao)

    return Image.fromarray(img_filtrada)

def aplicar_filtro_realce(imagem):
    img = np.array(imagem)

    if len(img.shape) == 3:
        altura, largura, canais = img.shape
    else:
        altura, largura = img.shape
        canais = 1
        img = img.reshape((altura, largura, 1))
    
    img_realcada = np.zeros_like(img)

    kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])

    for i in range(1, altura - 1):
        for j in range(1, largura - 1):
            for c in range(canais):
                regiao = img[i - 1:i + 2, j - 1:j + 2, c]

                valor_realcado = np.sum(regiao * kernel)

                img_realcada[i, j, c] = img[i, j, c] + valor_realcado
    
    return Image.fromarray(img_realcada)


def aplicar_filtro_media(imagem, tamanho_kernel):
    img = np.array(imagem)

    if len(img.shape) == 3:
        altura, largura, canais = img.shape
    else:
        altura, largura = img.shape
        canais = 1
        img = img.reshape((altura, largura, 1))

    img_suavizada = np.zeros_like(img)

    deslocamento = tamanho_kernel // 2

    # Aplicar o filtro de média
    for i in range(deslocamento, altura - deslocamento):
        for j in range(deslocamento, largura - deslocamento):
            for c in range(canais):
                regiao = img[i - deslocamento:i + deslocamento + 1, j - deslocamento:j + deslocamento + 1, c]

                img_suavizada[i, j, c] = np.mean(regiao)

    if canais == 1:
        img_suavizada = img_suavizada.reshape((altura, largura))

    return Image.fromarray(img_suavizada)

def obter_bordas(imagem, tamanho_kernel, limiar):
    if len(imagem.shape) == 3:
        imagem = np.mean(imagem, axis=2)
    
    img_filtrada = aplicar_filtro_media(imagem, tamanho_kernel)

    bordas = np.abs(imagem - img_filtrada)

    bordas[bordas < limiar] = 3
    bordas[bordas >= limiar] = 50

    return Image.fromarray(bordas.astype(np.uint8))

def reforcar_imagem(imagem, tamanho_kernel, fator_realce):
    if len(imagem.shape) == 3:
        imagem = np.mean(imagem, axis=2)

    img_suavizada = aplicar_filtro_media(imagem, tamanho_kernel)

    bordas = imagem - img_suavizada

    imagem_reforcada = imagem + fator_realce * bordas

    imagem_reforcada = np.clip(imagem_reforcada, 0, 255)

    return Image.fromarray(imagem_reforcada.astype(np.uint8))



imagem = Image.open('./exercicios/filtros/imagem-teste.jpeg').convert('L')

tamanho_kernel = 3
limiar = 30
fator_realce = 2.0

# resultado = aplicar_filtro_realce(imagem)
# resultado = aplicar_filtro_mediana(imagem, tamanho_kernel)

# resultado = obter_bordas(np.array(imagem), tamanho_kernel, limiar)
resultado = reforcar_imagem(np.array(imagem), tamanho_kernel, fator_realce)

# Salvar ou mostrar a imagem suavizada
resultado.save('./exercicios/filtros/resultados/resultado.jpg')
resultado.show()
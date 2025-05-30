## Quais seriam os melhores parâmetros para detectar bordas nas últimas figuras do Colab?
---

Usando suavização antes da detecção de bordas, como "cv2.GaussianBlur", ajuda a remover ruídos

Thresholds do Canny devem ser ajustados com base no contraste da imagem.
EX:
    cv2.Canny(imagem,50,150)
    cv2.Canny(imagem,100,200)

É importante testar visualmente diferentes combinações de parâmetros para encontrar o melhor resultado.

---

## Qual etapa de pre-processamento seria mais adequada?
---

GaussianBlur é altamente recomendada para suavizar a imagem antes de detectar as bordas. Reduz o ruído da imagem, o que evita que pequenos detalhes ou imperfeições sejam detectados como bordas falsas

EX:
    cv2.GaussianBlur(imagem,(5,5),0)

---

## Qual o tamanho do Kernel do filtro de suavização que mais contribui para a detecção de borda?
---

Tipos de tamanhos para kernel:
    (3,3) -> Suavização leve, mantém detalhes
    (5,5) -> Bom equilíbrio entre suavização e nitidez
    (7,7) -> Suavização forte, pode apagar detalhes finos

(5,5) é geralmente é o mais recomendado como ponto de inicio, mas dependendo do estado do trabalho,
pode ser alterado.

---

## Por que suavizar a imagem melhora a detecção de bordas?
---

1. Remove ruídos, que podem ser uma causa para a identificação de bordas falsas
2. Faz as bordas reais ficarem mais destacadas, deixando a transição entre regiões
    da imagem mais nitidas
3. Reduz a variação local de intensidade, o que ajuda o algoritmo a identificar as bordas verdadeiras

---


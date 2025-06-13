# Customizações EasyOCR

## Paramêtros usados para o `Reader()`

- `lang_list`: Uma lista com as linguagens que serão utilizadas para a detecção dos textos. Foi utilizado `[pt]`, pois o reconhecimento é para o português.
- `gpu`: Valor booleano que indica uso de gpu ou não. Foi utilizado o valor de `True`.

## Paramêtros usados no método `readtext()`

Os paramêtros utilizados foram escolhidos a partir de testes realizados em imagens do dataset.

- `Paragraph:` Combina o resultado como um paragrafo foi utilizado o valor `True`.
- `x_ths:` Controla o quão distante os caracteres podem estar separados no eixo x a ser considerado a mesma linha. O valor usado foi de `2.0`.
- `y_ths`: Funciona do mesmo jeito que o `x_ths`, porém para o eixo y. Foi utilizado o valor de `0.15`.
- `width_ths`: Considera o quão largo um grupo de caracteres podem ser horizontalmente para serem considerados uma outra *text box*. Foi utilizado o valor de `1.2`.
- `height_ths`: Funciona igual ao `width_ths`, porém para a altura. Foi utilizado o valor de `0.7`.
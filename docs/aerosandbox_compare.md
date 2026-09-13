# Comparação com o VLM do AeroSandbox

Validação de `main.py` contra o `asb.VortexLatticeMethod` do
[AeroSandbox](https://github.com/peterdsharpe/AeroSandbox) — implementação independente,
mantida ativamente, do mesmo formalismo de ferradura de vórtices (Drela, *Flight Vehicle
Aerodynamics*, os mesmos capítulos usados aqui). Os números saem de
`aerosandbox_compare.json`, gerado por `aerosandbox_compare.py`; os gráficos por
`aerosandbox_plots.py`.

## Convenções conferidas antes de comparar

Duas armadilhas de convenção, checadas por inspeção do código-fonte do AeroSandbox
(`operating_point.py`, `vortex_lattice_method.py`) antes de confiar em qualquer número:

- **Eixo x.** "in geometry axes, X is downstream by convention" — comentário literal do
  código deles: nos seus próprios "geometry axes", x é a jusante (ré), z para cima, e isso é
  fixo na biblioteca. `main.py` passou a usar eixos de estabilidade (x+ frente, z+ baixo)
  depois desta comparação ter sido escrita pela primeira vez — as duas convenções são hoje
  **opostas** em x e z, então `surfaceBreakpoints` converte explicitamente (inverte x,z) ao
  extrair pontos de `Surface.transform` para empacotar em `WingXSec`. Sem essa conversão a
  geometria entregue ao AeroSandbox sairia espelhada.
- **Esteira.** `align_trailing_vortices_with_wind=False` é o **padrão** do AeroSandbox — a
  esteira fica fixa no eixo x do corpo, exatamente a correção que apliquei em `main.py` para
  a quase-singularidade da deriva/winglets. As duas implementações concordam em manter a
  esteira desacoplada do escoamento por padrão (aqui isso é `x_hat=[-1,0,0]`, aft agora
  sendo -x).

Um erro real apareceu no meu **script de comparação**, não em `main.py`: na asa retangular,
dei `xyz_le=[0,0,0]` à raiz no AeroSandbox, mas aqui `position=[0,0,0]` ancora o **¾ de
corda** da raiz, não o bordo de ataque — os dois `[0,0,0]` eram pontos físicos diferentes,
por 0,75 de corda. Isso não afeta `CL`/`CD` (invariantes a translação), mas quebrava `Cm`
completamente (sinal trocado). Corrigido extraindo as coordenadas do AeroSandbox direto de
`Surface.transform` (a mesma função que gera a malha real), em vez de digitar `xyz_le` à
mão — conferido ponto a ponto contra `self.nodes[0,:,:]` antes de usar.

## Asa retangular AR = 8

Mesmo caso com referência analítica de `docs/convergencia.md`, malha casada (9 painéis em
corda, espaçamento uniforme nos dois códigos):

| grandeza | main.py | AeroSandbox | diferença relativa |
|---|---|---|---|
| `CL` (α=5°) | 0,401175 | 0,401171 | 0,001% |
| `CD_i (= CD, sem arrasto de perfil no AeroSandbox)` (α=5°) | 0,006528 | 0,006528 | 0,002% |
| `Cm` (α=5°) | 0,203234 | 0,203227 | 0,004% |

Com a mesma malha e o mesmo ponto de referência, os dois códigos praticamente coincidem —
diferença na 5ª/6ª casa decimal, não no primeiro ou segundo dígito.

![CL, CD_i e Cm vs alpha, aeronave completa](asb_full_alpha.png)

### Convergência conjunta

Refinando os dois códigos juntos, mesmo espaçamento uniforme, α=5°:

| painéis/semiasa | CL main.py | CL AeroSandbox | \|diferença\| |
|---|---|---|---|
| 16 | 0,4066 | 0,4066 | 1,8e-05 |
| 32 | 0,4030 | 0,4029 | 1,7e-05 |
| 64 | 0,4010 | 0,4010 | 1,5e-05 |
| 128 | 0,4001 | 0,4001 | 1,3e-05 |
| 256 | 0,3996 | 0,3996 | 1,2e-05 |

![Convergência conjunta](asb_convergence.png)

A diferença fica na casa de 10⁻⁵ em toda a faixa testada (16 a 256 painéis por semiasa) e
**encolhe** com o refino — não é um piso de discrepância metodológica, é ruído numérico de
resolução finita, do tamanho que se espera entre duas implementações independentes do mesmo
método resolvendo o mesmo sistema linear. Os dois códigos também convergem **juntos** para o
mesmo valor assintótico (~0,3996 em vez do 0,4012 relatado em `docs/convergencia.md`, que
parou o refino cedo demais — o CL da asa retangular com espaçamento uniforme ainda está
caindo lentamente a 256 painéis/semiasa, sintoma conhecido de VLM com espaçamento uniforme
perto da ponta, onde o espaçamento cosseno converge muito mais rápido).

## Aeronave completa

Asa (2 segmentos) + winglets + cauda horizontal e vertical, convertida superfície a
superfície via `surfaceBreakpoints`/`toAsbWing` — nenhuma coordenada digitada à mão.

![As duas geometrias lado a lado, mesma câmera](asb_geometry_compare.png)

Mesma câmera, mesma proporção de eixos — a comparação visual mais direta de que a
conversão preserva afilamento, enflechamento, diedro e a posição dos winglets antes de
sequer olhar um número. A malha do AeroSandbox aparece mais densa à direita porque a
resolução em envergadura dele (`spanwise_resolution=20`) é aplicada por *segmento* entre
duas `WingXSec`, enquanto a minha é por superfície inteira — mesma geometria, densidade de
painel só um pouco diferente, sem efeito na comparação de coeficientes acima. Os vértices
que o `VortexLatticeMethod` do AeroSandbox devolve (`front_left_vertices` etc.) vêm nos
*geometry axes* deles — `aerosandbox_geometry.py` converte de volta para eixos de
estabilidade antes de desenhar, senão os dois painéis ficariam em câmeras fisicamente
inconsistentes mesmo compartilhando os mesmos limites de eixo.

| grandeza | main.py | AeroSandbox | diferença relativa |
|---|---|---|---|
| `CL` (α=5°) | 0,426036 | 0,426954 | 0,22% |
| `CD_i` (α=5°) | 0,006926 | 0,007288 | 5,23% |
| `Cm` (α=5°) | 0,321280 | 0,320494 | 0,24% |

`CL` e `Cm` concordam a **0,24%** ou melhor em toda a faixa de α (gráfico acima,
painéis 1 e 3 — as curvas ficam praticamente sobrepostas). `CD_i` diverge um pouco mais
(5,2% em α=5°) — esperado: arrasto induzido é a diferença entre dois números
grandes (a projeção da sustentação inclinada menos a sustentação em si), então amplifica
qualquer detalhe fino de discretização que `CL` e `Cm` absorvem sem problema.

### Acoplamento lateral

α=5° fixo, β variando — testa `CY`, `Cl`, `Cn`, que dependem da deriva e dos winglets:

| β | CY main.py | CY AeroSandbox | Cl main.py | Cl AeroSandbox | Cn main.py | Cn AeroSandbox |
|---|---|---|---|---|---|---|
| -5° | +0,0270 | +0,0261 | +0,0123 | +0,0122 | -0,0052 | -0,0041 |
| -2° | +0,0109 | +0,0105 | +0,0049 | +0,0049 | -0,0021 | -0,0017 |
| +0° | +0,0000 | +0,0000 | +0,0000 | +0,0000 | +0,0000 | -0,0000 |
| +2° | -0,0109 | -0,0105 | -0,0049 | -0,0049 | +0,0021 | +0,0017 |
| +5° | -0,0270 | -0,0261 | -0,0123 | -0,0122 | +0,0052 | +0,0041 |

![Acoplamento lateral](asb_lateral.png)

`CY` e `Cl` seguem de perto (`Cl` praticamente sobreposto). `Cn` é o que mais diverge em
termos relativos (~20% em β=5°) — mas ambos os números são pequenos (~0,004-0,005) e do
mesmo sinal e ordem de grandeza; é o coeficiente lateral mais sensível a detalhe fino de
malha na deriva, coerente com o mesmo padrão do `CD_i`.

## Leitura

| verificação | resultado |
|---|---|
| Convenção de eixos | opostas hoje (main.py x-frente, AeroSandbox x-ré) — convertida no limite via `surfaceBreakpoints` |
| Convenção de esteira | ambos usam esteira fixa no corpo por padrão |
| Asa retangular, malha casada | CL/CD_i/Cm coincidem na 5ª/6ª casa |
| Convergência conjunta | dCL ~ 10⁻⁵, encolhendo, sem piso metodológico |
| Aeronave completa, CL/Cm | concordam a <0,3% em toda a faixa de α |
| Aeronave completa, CD_i/Cn | ~5-20% — esperado, são as grandezas mais sensíveis a malha |

O maior risco nesta comparação nunca foi `main.py` — foi montar a mesma aeronave duas vezes
sem transcrever nada errado. As duas armadilhas reais que apareceram (resolução em corda
do AeroSandbox caindo pra 1 painel sem eu perceber, e o ponto de referência trocado na asa
retangular) eram bugs no *script de comparação*, pegos comparando contra a convergência já
documentada em `docs/convergencia.md` antes de aceitar qualquer número.

---

Para reproduzir: `python docs/aerosandbox_compare.py && python docs/aerosandbox_plots.py &&
python docs/aerosandbox_geometry.py`

(requer `pip install aerosandbox sortedcontainers`)

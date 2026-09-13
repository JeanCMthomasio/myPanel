# Campo próximo × plano de Trefftz

`main.py` calcula as forças no **campo próximo**: Kutta–Joukowski em cada vórtice ligado
(eq. 6.42) e o arrasto induzido como a projeção da força no escoamento (eq. 6.50). Drela
apresenta na seção 6.4.4 a alternativa de **campo distante**, integrando no plano de
Trefftz (eq. 6.26 a 6.31), e afirma que ela é mais confiável "especially for the `D_i`
component, since it avoids the usual pressure-drag cancellation errors". Este documento
mede se isso aparece aqui. Os números saem de `trefftz.json`, gerado por `trefftz.py`.

O cálculo de Trefftz vive só no script de validação, não em `main.py`: é um caminho de
conferência, não uma funcionalidade do solver.

## Método

Longe a jusante a esteira vira um problema 2D. Cada coluna de painéis em corda deixa ali um
único par de fios, de intensidade igual à **soma** das circulações da coluna — os fios
internos de ferraduras vizinhas se cancelam. Entre os fios, a folha carrega o salto de
potencial `Δφ = Γ` da faixa (eq. 6.27/6.28). Com isso:

- `∇φ` no meio de cada faixa vem dos vórtices pontuais 2D (eq. 6.26);
- `CD_i` é a forma quadrática `∫ Δφ ∇φ·n̂ ds` (eq. 6.29);
- `CL` e `CY` saem lineares em `Δφ` (eq. 6.30/6.31), **já em eixos de vento** — a teoria de
  campo distante entrega a força perpendicular ao escoamento, então aplicar o `T_a` do
  campo próximo projetaria duas vezes.

Aquela projeção dupla foi de fato o primeiro erro: ela aparecia como um `cos²α` espúrio no
`e`, e foi assim que se revelou (ver "Duas armadilhas", abaixo).

## 1 · Varredura em α, asa retangular AR = 8

| α | CL perto | CL Trefftz | CD_i perto | CD_i Trefftz | e perto | e Trefftz |
|---|---|---|---|---|---|---|
| 1° | 0,08065 | 0,08065 | 0,000263 | 0,000263 | 0,9840 | **0,9840** |
| 2° | 0,16124 | 0,16128 | 0,001051 | 0,001052 | 0,9841 | **0,9840** |
| 5° | 0,40220 | 0,40277 | 0,006535 | 0,006560 | 0,9849 | **0,9840** |
| 8° | 0,64083 | 0,64315 | 0,016564 | 0,016727 | 0,9864 | **0,9840** |
| 12° | 0,95305 | 0,96081 | 0,036514 | 0,037330 | 0,9898 | **0,9840** |

Este é o resultado central. Numa teoria linear com planta fixa, `CL ∝ α` e `CD_i ∝ α²`,
então a eficiência de envergadura `e = CL²/(πAR·CD_i)` **não pode depender de α**. O campo
distante obedece isso exatamente: 0,9840 nas cinco incidências, sem variar um dígito. O
campo próximo sobe de 0,9840 para 0,9898 — uma dependência espúria de ordem α² que só pode
vir do cancelamento imperfeito entre as parcelas de força que o campo próximo soma.

É exatamente a falha que Drela aponta, medida aqui. Em compensação, a magnitude é pequena
na faixa útil: até α = 8° os dois métodos concordam em 0,4% no `CL` e 1% no `CD_i`.

## 2 · Convergência de malha (α = 5°)

| nc × ns | N | CD_i perto | CD_i Trefftz | e perto | e Trefftz |
|---|---|---|---|---|---|
| 2 × 10 | 40 | 0,006557 | 0,006582 | 1,0221 | 1,0210 |
| 4 × 20 | 160 | 0,006548 | 0,006573 | 0,9971 | 0,9961 |
| 8 × 40 | 640 | 0,006535 | 0,006560 | 0,9849 | 0,9840 |
| 12 × 60 | 1440 | 0,006529 | 0,006554 | 0,9809 | 0,9800 |
| 16 × 80 | 2560 | 0,006526 | 0,006551 | 0,9789 | 0,9780 |

Aqui o campo distante **não** é mais convergido: a diferença entre os dois fica em 0,4%
constante em toda a série, e os dois descem juntos. Nas malhas grosseiras ambos violam o
limite elíptico (`e > 1`), o artefato já conhecido de espaçamento uniforme perto da ponta
documentado em `convergencia.md` — não é privilégio de um método. A vantagem do campo
distante, portanto, é de **consistência em α**, não de convergência em malha.

## 3 · Aeronave completa

| α | CL perto | CL Trefftz | CD_i perto | CD_i Trefftz | dif. CD_i | Cm |
|---|---|---|---|---|---|---|
| 2° | 0,19123 | 0,19106 | 0,001385 | 0,001407 | 1,5% | 0,11461 / — |
| 5° | 0,47775 | 0,47715 | 0,008683 | 0,008776 | 1,1% | 0,28284 / — |
| 8° | 0,76229 | 0,76193 | 0,022191 | 0,022376 | 0,8% | 0,44501 / — |

Com deriva, winglets e cauda, `CL` concorda em 0,1% e `CD_i` em ~1%. Em derrapagem
(α = 5°, β = ±5°) o `CY` fica em +0,05238 contra +0,05093, 2,8% — o coeficiente lateral é o
mais sensível, coerente com o padrão já visto na comparação com o AeroSandbox.

A coluna do `Cm` está vazia de propósito: **o plano de Trefftz não dá momentos**. É a
ressalva final do próprio texto do Drela — "this Trefftz plane force calculation method
gives only the total forces. Equations (6.24) and (6.25) must be used for moments, and also
for forces on the individual surfaces". Isso sozinho decide a arquitetura: como o objetivo
aqui é estabilidade e controle, e `Cm`, `Cl`, `Cn` são metade dos resultados, o campo
próximo tem que existir de qualquer jeito. O campo distante só poderia ser um complemento
para `CD_i`, nunca um substituto.

## 4 · O campo distante resolve a diferença com o AeroSandbox?

`aerosandbox_compare.md` registra `CD_i` 5,2% abaixo do AeroSandbox na aeronave completa —
a maior discrepância daquele estudo. Se o campo próximo fosse o culpado, o campo distante
deveria encostar mais perto. Na mesma aeronave, α = 5°:

| | CD_i | desvio do AeroSandbox |
|---|---|---|
| campo próximo | 0,006926 | 4,97% |
| Trefftz | 0,006941 | 4,76% |
| AeroSandbox | 0,007288 | — |

**Não resolve.** Os dois métodos ficam praticamente no mesmo lugar, a 5% do AeroSandbox.
Isso é informativo: descarta o cálculo de força como origem daquela diferença e reforça a
leitura que já estava lá — é diferença de discretização entre os dois códigos (a resolução
em envergadura do AeroSandbox é aplicada por segmento, a minha por superfície), não de
formulação.

## Duas armadilhas, ambas no script deste estudo

Nenhuma das duas estava em `main.py`; as duas custaram caro o suficiente para valer o
registro.

1. **Projeção dupla.** A força de Trefftz já sai perpendicular ao escoamento; passá-la pelo
   `T_a` do campo próximo a projetava de novo. O sintoma foi `e` variando exatamente com
   `cos²α` (0,9828 / 0,9765 / 0,9649 em α = 2/5/8°, todos dando 0,9840 ao dividir por
   `cos²α`) — a regularidade perfeita do fator é que denunciou a causa.
2. **Tangente 3D em vez da projetada.** A folha vive no plano de Trefftz, então a tangente e
   o comprimento de cada faixa têm que sair da projeção de `B − A` em (y,z). Usando a
   tangente 3D entra a componente x do enflechamento da linha de ¼ de corda, que ali não
   existe. Numa asa **retangular isso não aparece** (0,14%); na asa enflechada e afilada do
   `__main__` valia **10,5%** de erro no `CL`. O que identificou o erro como estrutural foi
   refinar a malha e a diferença **não** cair (10,49% → 10,97%): discretização melhora com
   refino, erro de geometria não.

## Resumo

| verificação | resultado |
|---|---|
| 1. Varredura em α | `e` de Trefftz constante (0,9840); campo próximo deriva até 0,9898 em 12° |
| 2. Convergência | diferença de 0,4% constante; nenhum dos dois converge mais rápido |
| 3. Aeronave completa | `CL` a 0,1%, `CD_i` a ~1%, `CY` a 2,8%; Trefftz não dá momento nenhum |
| 4. Contra o AeroSandbox | não fecha a diferença de 5% — a causa não é o cálculo de força |

**Conclusão prática.** A crítica do Drela ao campo próximo é real e mensurável, mas neste
código ela vale menos de 1% até α = 8°, enquanto o campo distante não fornece `Cm`, `Cl`,
`Cn` nem a decomposição por superfície. Não há motivo para trocar a formulação de
`main.py`; o valor do plano de Trefftz aqui é ser uma verificação independente do `CD_i`,
que é justamente o coeficiente mais frágil do método.

---

Para reproduzir: `python docs/trefftz.py`

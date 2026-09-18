# Superfícies de comando

Validação da implementação de comandos em `main.py`: rotação das normais em torno da
linha de charneira (Drela), montagem do ganho global por painel, e derivadas de controle
(`∂C/∂δ`) por diferenciação automática. Os números deste documento saem de
`controles.json`, gerado por `controles.py`.

## Método

O comando não move a treliça de vórtices — só rotaciona a normal de cada painel em torno
de `hinge_axis = (vortex_b − vortex_a)/|·|`, que já é paralelo à linha de charneira (uma
articulação a x/c constante segue o mesmo enflechamento da malha) e perpendicular à normal
(medido `|dl·n| < 7,5×10⁻⁹`). Com isso a fórmula de Rodrigues colapsa sem termo residual:

```
n' = n·cos(θ) + (ĥ × n)·sin(θ)
```

`θ` por painel é a soma ponderada das deflexões que tocam aquele painel —
`self.control_gain.T @ deg2rad(deltas)` — onde `control_gain` é uma matriz `(C, N)` com uma
linha por comando (ordem em `self.control_names`) e uma coluna por painel da aeronave,
zero nos painéis que o comando não toca. Cada entrada é o produto de duas máscaras:

- **em corda**: fração do painel que fica atrás da charneira, `clip((x−hinge)/Δx, 0, 1)` —
  o painel atravessado pela charneira recebe ganho fracionário em vez de tudo-ou-nada;
- **em envergadura**: 1 dentro da faixa `span=(s0,s1)` (fração da semi-envergadura), com o
  sinal invertido no lado espelhado quando `mode='antisymmetric'` (aileron), mesmo sinal
  dos dois lados quando `mode='symmetric'` (flap, profundor).

Nem o kernel de Biot–Savart (`self.influence_collocation`, `self.influence_aero`) nem a
força de Kutta–Joukowski dependem da normal — só a AIC depende. Por isso o único passo
caro, o kernel `(N,N,3)`, fica em `computeInfluences` e é montado uma vez, com a malha;
`coefficientsAt(alpha, beta, deltas)` só reprojeta a normal e resolve, encadeando
`deflect` → `computeAIC` → `solveSystem` → `computeCoefficients`. Cada um desses passos
**devolve** seu resultado em vez de escrever em `self`, e é isso que deixa `coefficientsAt`
passar por `jax.jacfwd` (ver `controlDerivatives`): uma atribuição `self.AIC = ...` dentro
do traço deixaria um tracer morto pendurado no objeto depois que a AD retornasse.

## 1 · Regressão

`coefficientsAt(5.0, 0.0, None)` contra `coefficientsAt(5.0, 0.0, zeros)` na mesma aeronave,
α=5°: `deflect(None)` devolve `self.normals` bit a bit (`cos(0)=1`, `sin(0)=0` exatos,
conferido), então o caso sem deflexão é exatamente o caso geral com δ=0 — não um caminho de
código à parte. Os dois caminhos montam `V_bar`/`T_a` com a mesma expressão, na mesma
aritmética — não sobra nenhuma conversão de precisão entre eles para explicar diferença
nenhuma, e não sobra: as seis saídas batem bit a bit.

| coeficiente | `deltas=None` | `deltas=zeros` | \|diferença\| |
|---|---|---|---|
| `CD_i` | 0,007151833 | 0,007151833 | 0,0e+00 |
| `CY` | 4,7e-10 | 4,7e-10 | 0,0e+00 |
| `CL` | 0,414442092 | 0,414442092 | 0,0e+00 |
| `Cl` | 6,8e-09 | 6,8e-09 | 0,0e+00 |
| `Cm` | 0,317727476 | 0,317727476 | 0,0e+00 |
| `Cn` | -5,0e-10 | -5,0e-10 | 0,0e+00 |

Diferença máxima: **0,0e+00** — idêntico bit a bit, não só dentro do ruído de arredondamento.

## Simetria par/ímpar

A primeira versão deste plano previa "`Cl≠0` com `CL,CY,Cm,Cn≈0`" para o aileron sozinho —
impreciso, corrigido aqui antes de rodar o teste. A derivação correta vem de espelhar a
aeronave (`Y → −Y`): isso mapeia uma deflexão em modo antissimétrico na sua própria negativa,
`δ → −δ`. Forças são vetores verdadeiros (a componente ao longo do eixo de espelhamento
inverte); momentos são pseudovetores (as componentes *perpendiculares* ao eixo invertem, a
*paralela* não). Combinando os dois:

| | comportamento em δ | consequência |
|---|---|---|
| `CY`, `Cl`, `Cn` | ímpares | livres para não-zerar, ordem linear |
| `CL`, `CD_i`, `Cm` | pares | zero na ordem linear — só resíduo O(δ²) |

Vale tanto para o aileron (modo antissimétrico, par de superfícies espelhadas) quanto para
um leme — um fin único centrado em Y=0 é simétrico como *forma*, mas defletido desloca
material para um lado só, então `espelho(estado(δ)) = estado(−δ)` se aplica igual.

## 2 · Aileron sozinho

α=β=0°, só o aileron da asa defletido em 5,0°:

| par, ≈0 | valor | | ímpar, livre | valor |
|---|---|---|---|---|
| `CL` | -0,000009 | | `CY` | -0,001334 |
| `CD_i` | 0,000704 | | `Cl` | -0,018774 |
| `Cm` | 0,000044 | | `Cn` | -0,000314 |

`CL` sai essencialmente zero (`−9×10⁻⁶`) e `Cl` domina claramente as demais — o par certo
de sinais para um comando que existe para gerar rolamento. `CY` e `Cn` não-zerarem não é
falha: pela tabela acima, nada os obriga a ficar pequenos, e fisicamente correspondem à
guinada adversa/proversa clássica de aileron.

## 3 · Leme

Aeronave separada (asa reta + um fin único em Y=0 com `rudder`), leme em
10,0°:

| coeficiente | valor |
|---|---|
| `CD_i` | 0,001083 |
| `CY` | -0,030673 |
| `CL` | -0,000144 |
| `Cl` | -0,004780 |
| `Cm` | 0,001661 |
| `Cn` | 0,007711 |

`CY` e `Cn` dominam — força lateral e guinada, o par que um leme deveria produzir. `Cl` sai
não-nulo mas secundário (força lateral do fin atuando fora do eixo de rolamento, braço em
Z) — livre pela mesma tabela par/ímpar, não indício de erro. `CL` fica em `−0,0001`, ≈0
como esperado.

## 4 · Efetividade de flap × teoria de perfil fino

Asa reta de alto AR, flap de envergadura cheia em x/c=0,75. A teoria de
perfil fino dá

```
∂CL/∂δ ÷ ∂CL/∂α = (π − θh + sin θh)/π,   cos θh = 1 − 2·(x_h/c)
```

que para x_h/c=0,75 vale **0,6090**. `∂CL/∂δ` vem de
`controlDerivatives` (AD exata); `∂CL/∂α` de diferença central em α (independente de δ):

| painéis em corda | N | razão VLM | desvio da teoria |
|---|---|---|---|
| 4 | 240 | 0,4752 | 22,0% |
| 6 | 400 | 0,5554 | 8,8% |
| 9 | 640 | 0,5848 | 4,0% |
| 14 | 1040 | 0,5953 | 2,2% |
| 20 | 1520 | 0,6026 | 1,1% |

Convergência monotônica e limpa — de 22% a 1,1% de desvio conforme a corda refina, exatamente
onde o ganho fracionário na charneira se justifica (célula atravessada pela linha de comando).
O resíduo de ~1% no caso mais fino é esperado: a asa de teste tem AR alto mas finito, e a
teoria pressupõe AR infinito (escoamento 2D).

## 5 · AD × diferença finita

Em vez de confiar num único `h`, varri a largura do passo: a diferença finita central troca
erro de truncamento (h grande) por cancelamento numérico (h pequeno), e o erro relativo
desenha um V contra `h` — o vértice fica onde os dois efeitos se cruzam, e a posição depende
da precisão da aritmética.

O flap deste teste é simétrico e α=β=0, então `CD_i`, `CY`, `Cl`, `Cn` são analiticamente
zero (mesmo argumento par/ímpar da seção anterior, com os papéis trocados: comando em modo
simétrico deixa os laterais em zero, não `CL`/`Cm`). `J_ad` confirma isso — saem em ruído de
arredondamento, não sinal real. A métrica de erro relativo restringe-se a `CL` e `Cm`, as
únicas com derivada de verdade; dividir ruído por ruído nos outros componentes só mediria o
piso de precisão, não a AD.

| h (graus) | erro relativo máximo (CL, Cm) | anterior, em float32 |
|---|---|---|
| 1,0000 | 1,02e-04 | 1,02e-04 |
| 0,1000 | 1,02e-06 | 1,22e-06 |
| 0,0100 | 1,02e-08 | 8,12e-08 |
| 0,0010 | 1,02e-10 | 2,37e-07 |
| 0,0001 | 1,02e-12 | 2,11e-07 |

Com o solver em float64 (`jax_enable_x64`) a queda é **monotônica e exatamente de segunda
ordem**: cada 10× em `h` dá 100× menos erro, ao longo de oito ordens de grandeza. O vértice
do V existe, mas foi empurrado para fora desta faixa de `h` — em float32 ele aparecia em
h≈0,01° com piso de 8,1e-08 (coluna à direita), e era esse cancelamento que limitava a
comparação.

A concordância limpa em 1,02e-12 é a confirmação mais forte da AD: um erro que segue a lei
de segunda ordem da própria diferença finita, sem piso de precisão contaminando a medida.

## Resumo

| verificação | resultado |
|---|---|
| 1. Regressão | max\|dif\| = 0,0e+00 — bit a bit (δ=0 é o caso geral, não outro caminho) |
| 2. Aileron | `Cl` domina, `CL`≈0, `CY`/`Cn` livres e pequenos — sinais corretos |
| 3. Leme | `CY`/`Cn` dominam, `CL`≈0 |
| 4. Flap × teoria | convergência monotônica, 1,1% de desvio no caso mais fino |
| 5. AD × dif. finita | queda monotônica de 2ª ordem até 1,02e-12 (CL,Cm) em float64 |

---

Para reproduzir: `python docs/controles.py`

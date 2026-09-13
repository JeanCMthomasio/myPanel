# Impacto da discretização nos resultados

Estudo de convergência do `main.py`. Todas as tabelas e gráficos deste documento saem
de `study.json`, gerado por `study.py`; os gráficos por `plots.py`.

> **Nota sobre esta atualização.** A versão anterior deste estudo foi recalculada porque
> a esteira das ferraduras passou a ficar fixa no eixo x do corpo em vez de acompanhar o
> escoamento (`x_hat` desacoplado de `V_bar` em `computeInfluences`/`computeCoeficients`).
> A mudança corrige uma quase-singularidade: numa superfície vertical de baixo
> enflechamento como a deriva, deixar a esteira girar com α realinha a cada grau qual
> coluna de vórtices fica quase colinear com ela, colapsando o denominador `|a|-a·x̂` da
> eq. 6.33. **Isso invalida os números de `cond(AIC)` da seção "Aeronave completa" da
> versão anterior** — medi retroativamente o modelo antigo nas mesmas malhas: `cond(AIC)`
> ficava em **9,7×10⁸** constante em toda a série de refino da asa (dominado pela deriva,
> não pela asa) e chegava a **2,3×10¹¹** no caso de cauda mais refinado — em float32 isso
> não deixa nenhum dígito de precisão real. As conclusões qualitativas sobre convergência
> se mantêm (ver abaixo), mas os valores numéricos da seção da aeronave completa mudaram.
> O caso da asa retangular sozinha (seção seguinte) não tem superfície vertical e nunca
> foi afetado por essa quase-singularidade; ainda assim os valores de `CL` e `CD_i` mudam
> ligeiramente porque o modelo físico da esteira também mudou — são dois VLM lineares
> legítimos e distintos, que só coincidem exatamente em α=0.

## Caso de referência: asa retangular AR = 8

Corda 1, envergadura 8, α = 5°, sem enflechamento nem diedro. Escolhida porque tem
resposta conhecida para comparar:

| grandeza | referência |
|---|---|
| `CL_α` | ≈ 4,70 / rad (VLM / Weissinger) |
| `x_cp` | −0,5000 (¼ de corda; o bordo de ataque está em −0,75) |
| `e` | < 1 — igual a 1 só para carregamento elíptico |

### Refino em corda (envergadura fixa em 24 painéis/semiasa)

| corda×env | N | CL | CL_α | CD_i | e | x_cp | cond(AIC) |
|---|---|---|---|---|---|---|---|
| 1×24 | 48 | 0,4020 | 4,606 | 0,00646 | 0,995 | -0,4988 | 14,8 |
| 2×24 | 96 | 0,4037 | 4,626 | 0,00653 | 0,994 | -0,5045 | 13,7 |
| 4×24 | 192 | 0,4041 | 4,631 | 0,00654 | 0,993 | -0,5061 | 13,9 |
| 8×24 | 384 | 0,4042 | 4,632 | 0,00655 | 0,993 | -0,5065 | 14,8 |
| 16×24 | 768 | 0,4042 | 4,632 | 0,00655 | 0,993 | -0,5066 | 16,0 |
| 32×24 | 1536 | 0,4042 | 4,632 | 0,00655 | 0,993 | -0,5066 | 17,4 |

### Refino em envergadura (corda fixa em 8 painéis)

| corda×env | N | CL | CL_α | CD_i | e | x_cp | cond(AIC) |
|---|---|---|---|---|---|---|---|
| 8×4 | 64 | 0,4251 | 4,872 | 0,00654 | 1,099 | -0,5045 | 2,9 |
| 8×8 | 128 | 0,4135 | 4,738 | 0,00658 | 1,034 | -0,5058 | 5,1 |
| 8×16 | 256 | 0,4066 | 4,660 | 0,00656 | 1,003 | -0,5063 | 9,9 |
| 8×24 | 384 | 0,4042 | 4,632 | 0,00655 | 0,993 | -0,5065 | 14,8 |
| 8×40 | 640 | 0,4022 | 4,609 | 0,00653 | 0,985 | -0,5066 | 24,5 |
| 8×60 | 960 | 0,4012 | 4,597 | 0,00653 | 0,981 | -0,5066 | 36,7 |

![Convergência do CL](conv_cl.png)

**A corda converge quase instantaneamente.** Com um único painel em corda o `CL` já está
a 0,20% do valor final e o `x_cp` a
1,54%. De 8 para 32 painéis nada muda na quarta casa —
você quadruplica N e paga N² de memória por zero.

**A envergadura é onde o erro mora.** Com 4 painéis por semiasa o `CL` erra
6,0%, e só cai abaixo de 0,3% a partir de 24.

![Eficiência de envergadura](conv_e.png)

**O `CD_i` é o mais lento, e dá resposta impossível quando subresolvido.** Com 4 e 8 painéis
em envergadura o `e` sai **1,099** e **1,034** — fisicamente impossível numa asa
planar. É a região da ponta, onde o carregamento despenca, mal resolvida. E mesmo com 60
painéis o `e` ainda está descendo (0,985 → 0,981): se o arrasto induzido importa,
ele exige bem mais malha que a sustentação. Essa parte do estudo é praticamente idêntica à
versão anterior — `e` depende da distribuição de carregamento ao longo do vão, não da
direção da esteira, então o modelo antigo e o novo concordam aqui.

O convergido dá `CL_α = 4,597/rad` e `x_cp = -0,5066`. O resíduo de
0,0066 no `x_cp` é o efeito de AR finito, não numérico.

## Aeronave completa

Asa de dois segmentos + empenagem horizontal e vertical, como no `__main__`. CG na origem,
ponto neutro medido por `dCm/dCL` entre 0° e 5°. A coluna `cond(AIC)` é o maior valor entre
os dois ângulos — com a esteira fixa no corpo ela fica estavelmente baixa em toda a série,
ao contrário do que o modelo antigo dava (ver nota no topo do documento).

### Refinando a asa (cauda fixa em 9×14)

| painéis na asa | N total | CL (5°) | x_np | CD_i | cond(AIC) |
|---|---|---|---|---|---|
| 90 | 468 | 0,4180 | 0,2006 | 0,00709 | 102,1 |
| 220 | 598 | 0,4163 | 0,2007 | 0,00722 | 67,2 |
| 504 | 882 | 0,4149 | 0,2001 | 0,00729 | 50,6 |
| 900 | 1278 | 0,4142 | 0,1998 | 0,00733 | 41,0 |
| 1562 | 1940 | 0,4137 | 0,1995 | 0,00735 | 52,4 |

### Refinando a cauda (asa fixa em 9×10 + 9×27)

| painéis na cauda | N total | CL (5°) | x_np | CD_i | cond(AIC) |
|---|---|---|---|---|---|
| 12 | 666 | 0,4163 | 0,2137 | 0,00732 | 39,0 |
| 48 | 720 | 0,4155 | 0,2059 | 0,00732 | 26,4 |
| 140 | 858 | 0,4150 | 0,2021 | 0,00732 | 26,3 |
| 252 | 1026 | 0,4148 | 0,2003 | 0,00732 | 41,1 |
| 572 | 1506 | 0,4147 | 0,1986 | 0,00732 | 91,8 |

![Ponto neutro](conv_np.png)

**O ponto neutro continua governado pela cauda, não pela asa** — a conclusão qualitativa
da versão anterior se mantém, mesmo com o modelo de esteira corrigido e os valores
absolutos diferentes. Quadruplicar os painéis da asa move o `x_np` em
0,0012 m; refinar a cauda move 0,0152 m — **12,9× mais sensível**, apesar de a asa ter
cerca de 10× mais painéis. (A razão era ~42× no estudo anterior; a diferença vem do
`cond(AIC)` daquela série estar ele próprio dependente da malha da cauda de um jeito
artificial, por causa da quase-singularidade — a razão de agora é a confiável.)

Faz sentido: a contribuição da cauda para `dCm/dα` depende da inclinação de sustentação dela
própria, e superfície pequena precisa de resolução própria. Com a cauda em 2×3 a margem
estática erra 1,5% da MAC.

## Custo

| N | kernel (N,N,3) | AIC (N,N) | `np.linalg.solve` |
|---|---|---|---|
| 960 | 11 MB | 4 MB | 0,0047 s |
| 1920 | 44 MB | 15 MB | 0,0104 s |
| 3840 | 177 MB | 59 MB | 0,0284 s |

O `solve` escala perto de N^2,4 (BLAS bom) e é barato. **O limitante é memória, não tempo** —
e são dois kernels `(N,N,3)`, mais `bound`, `leg_a` e `leg_b` como arrays separados antes de
somar, então na prática multiplique por ~5. Esses números não mudam com a correção da
esteira — são puramente função do tamanho de N.

## Recomendação

A asa do `__main__` está em `[(10,11),(10,31)]` — 9 painéis em corda e 40 em envergadura por
semiasa. A envergadura está bem dimensionada; a corda tem o dobro do necessário. Essa leitura
não muda com a correção da esteira: é uma questão de densidade de malha, não do modelo
físico usado para resolver o sistema.

Testei a malha reduzida contra a atual na aeronave completa (não estava nas séries de refino
acima, que variam asa e cauda em separado):

| superfície | hoje | sugerido | efeito isolado em x_np |
|---|---|---|---|
| asa | `[(10,11),(10,31)]` → 720 painéis | `[(5,11),(5,31)]` → 320 | −0,0012 m |
| cauda h. | `[(10,15)]` → 252 | `[(5,23)]` → 176 | −0,0018 m |
| cauda v. | `[(10,15)]` → 126 | `[(5,23)]` → 88 | (incluído acima) |

Os dois efeitos são quase aditivos: com as três reduções juntas, `x_np` vai de 0,1998 para
0,1969 (desce de 0,200 para 0,197 na 3ª casa — **não fica inalterado**, ao contrário do que a
versão anterior deste documento dizia sem ter testado a combinação real). `CL` sim fica estável
na 4ª casa (0,4147 → 0,4145). O deslocamento de 0,0029 m é ~0,3% da MAC — pequeno, mas real, e
vale saber que existe antes de comparar resultados entre as duas malhas.

Total cai de 1098 para 584 painéis, e a memória dos kernels (vai com N²) cai a ~28%. Se ~0,3%
da MAC no ponto neutro for tolerável para o seu uso, a redução compensa; se não for, mantenha
a cauda na malha atual e reduza só a asa (efeito isolado: −0,0012 m, menos da metade).

Se o arrasto induzido for a grandeza de interesse, inverta a prioridade: `CD_i` ainda não
converge com 60 painéis em envergadura, enquanto `CL` e `x_np` já estão estáveis com 24.

---

Para reproduzir: `python docs/study.py && python docs/plots.py`

# 🃏 Blackjack – TD Methods (SARSA & Q-learning)

Ovaj repozitorijum sadrži implementaciju **Blackjack** problema (Gym‑style okruženje) sa fokusom na **Temporal Difference (TD)** metode učenja, konkretno **SARSA** i **Q‑learning**. Projekat je namenjen učenju i eksperimentisanju sa RL algoritmima na klasičnom kartičnom problemu.

---

##  Sadržaj projekta

* Implementacija Blackjack okruženja (karte, špil, pravila)
* Definicija stanja, akcija i politika
* **SARSA (on‑policy TD control)**
* **Q‑learning (off‑policy TD control)**
* Vizualizacija naučenih politika
* Poređenje ponašanja algoritama

Glavni notebook:

* `blackjack_TD_methods.ipynb`

---

## Opis problema – Blackjack

Cilj agenta je da nauči optimalnu politiku igranja Blackjacka:

* **Stanja (State)** tipično uključuju:

  * Trenutni zbir karata igrača
  * Vidljivu kartu dilera
  * Informaciju da li igrač ima *usable ace*

* **Akcije (Actions)**:

  * `HIT` – povuci još jednu kartu
  * `HOLD / STAND` – završi potez

* **Nagrade (Rewards)**:

  * Pobeda: `+1`
  * Poraz: `-1`
  * Nerešeno: `0`

Epizoda se završava kada igrač *bust‑uje*, stane, ili se igra završi poređenjem sa dilerom.

---

## Korišćeni algoritmi

### SARSA (State–Action–Reward–State–Action)

* **On‑policy** TD metoda
* Učenje se vrši na osnovu politike koja se trenutno izvršava

Karakteristike:

* Stabilnije ponašanje
* Konzervativnija politika

---

### Q‑learning

* **Off‑policy** TD metoda
* Uči optimalnu politiku nezavisno od ponašajne

Karakteristike:

* Agresivnije učenje
* Brže konvergira ka optimalnoj politici

---

## Vizualizacija politika

Nakon učenja, politike se vizualizuju u obliku **heatmapa / tabela** koje prikazuju:

* Kada je optimalno `HIT`
* Kada je optimalno `HOLD`

Odvojeno za:

* Stanja sa *usable ace*
* Stanja bez *usable ace*

Ovo omogućava intuitivno poređenje sa poznatim optimalnim Blackjack strategijama.

---

## Pokretanje projekta

1. Kloniraj repozitorijum
2. Pokreni Jupyter Notebook:

```bash
jupyter notebook blackjack_TD_methods.ipynb
```

3. Izvršavaj ćelije redom i posmatraj učenje i politike

---

## Tehnologije

* Python 3
* NumPy
* Matplotlib
* Jupyter Notebook

---

## Cilj projekta

Ovaj projekat je edukativnog karaktera i ima za cilj:

* Razumevanje razlike između **on‑policy** i **off‑policy** TD metoda
* Prakticno učenje Reinforcement Learning‑a
* Analizu ponašanja algoritama na jednostavnom, ali nelinearnom problemu

---

## Napomene

* Zarad ispravne konvergencije finalne politike koristio se majorty vote akcija u svim stanjima za nekoliko politika
* Pravila su promenjena za rad sa realnim (koliko toliko) BJ pravilima
* Profesorovi plotovi nisu doradjivani i politike su izbrisane ali ostatak implementacije je tu

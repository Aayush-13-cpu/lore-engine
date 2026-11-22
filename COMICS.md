# Comics to get, video by video

**The rule:** before rendering a video, get the comics its story comes from. Real panels tell ~40% of the story;
AI draws the other ~60% (cosmic events, places and moments the comics never drew). Never use piracy sites:
one DMCA notice can take down this repo or strike the channel.

## How to do it yourself for any topic

1. **Find the source issues.** Look the character or event up on
   [Marvel Fandom](https://marvel.fandom.com) / [DC Fandom](https://dc.fandom.com) (section *First appearance*,
   *History*, the story-arc pages) or [League of Comic Geeks](https://leagueofcomicgeeks.com) (search the arc name).
   Pick the 3-6 issues where the events in your script actually happen. Fewer issues, more matching panels.
2. **Get them legally.** Marvel Unlimited (Marvel), DC Universe Infinite (DC), Kindle / ComiXology / publisher
   stores, Humble Bundle (DRM-free bundles), or your own print copies. Screenshot or photograph each page flat.
   Public-domain Golden Age comics are free: [Digital Comic Museum](https://digitalcomicmuseum.com),
   [Comic Book Plus](https://comicbookplus.com), Wikimedia Commons.
3. **Store them privately.** In your private `lore-assets` repo: `comics/<id>/01.jpg, 02.jpg, ...`
   (or `comics/<id>.cbz` / `.pdf`). Never in this public repo.
4. **List them** in the sheet's **Sources** tab: `source_id` = `<ID>` (e.g. `FF48`), `location` = `comics/ff48`,
   `video_id` = the video.
5. **Storyboard:** `./lore.sh storyboard VID00X` gives every page with numbered panels + their lettering.
6. **Match panels to lines:** in **Scenes → page**, write `FF48:5#2` (issue FF48, page 5, panel 2) on the line
   that panel illustrates. Consecutive lines on the same page glide panel to panel. Aim for ~40% of lines.
   Optional: **Characters → ref_page** `FF48:12@0.1,0.2,0.5,0.9` cuts the character's look from a real panel so
   the AI shots stay on-model.
7. `./lore.sh draft VID00X` to check framing, then `./lore.sh render VID00X`.

## The list

`ID` is what to name the folder (`comics/<id>`). **verify** = double-check the issue number on Fandom before buying.

### VID001 · Galactus: the complete lore
| ID | Comic | Chapter |
|---|---|---|
| ff48 | Fantastic Four #48 (1966) | The Coming of Galactus: Watcher's warning, Surfer finds Earth |
| ff49 | Fantastic Four #49 (1966) | Galactus arrives, Alicia and the Surfer |
| ff50 | Fantastic Four #50 (1966) | Ultimate Nullifier, Galactus leaves, Surfer exiled |
| ss1 | Silver Surfer #1 (1968) | Norrin Radd's deal on Zenn-La |
| ff262 | Fantastic Four #262 (1984) | The Trial of Galactus |
| thor169 | Thor #169 (1969) — optional | Galan of Taa's origin |

### The Ultimate Nullifier explained
| ID | Comic | Use |
|---|---|---|
| ff50 | Fantastic Four #50 (1966) | First appearance |
| ff48, ff49 | Fantastic Four #48-49 (1966) | Setup |

### Every herald of Galactus, ranked
| ID | Comic | Herald |
|---|---|---|
| ff48 | Fantastic Four #48 (1966) | Silver Surfer |
| ff120 | Fantastic Four #120 (1972) | Air-Walker |
| thor225 | Thor #225 (1974) | Firelord |
| ff211 | Fantastic Four #211 (1979) | Terrax |
| ff244 | Fantastic Four #244 (1982) | Nova (Frankie Raye) |
| — | Morg, Stardust: Silver Surfer vol. 3 (1990s) — **verify** issues | |

### Homelander / Black Noir (The Boys, comics vs show)
| ID | Comic | Use |
|---|---|---|
| boys1 | The Boys #1 (2006) | The Seven, Homelander introduced |
| boys60-72 | The Boys #60-72 (2011-2012) — **verify** exact issues | Final arcs, Black Noir twist, ending |

### Compound V
| ID | Comic | Use |
|---|---|---|
| boys1-30 | The Boys #1-30 (2006-2009) — **verify** which issues explain it | Origin and Vought |

### The Viltrumite Empire / Thragg / how Invincible ends
| ID | Comic | Use |
|---|---|---|
| inv1 | Invincible #1 (2003) | Mark and Omni-Man |
| inv11-12 | Invincible #11-12 (2004) — **verify** | Omni-Man's betrayal |
| inv61-78 | Invincible #61-78 (Viltrumite War) — **verify** | The Empire, Thragg |
| inv139-144 | Invincible #139-144 (2017-2018) | The ending |

### Knull
| ID | Comic | Use |
|---|---|---|
| venom1-6 | Venom (2018) #1-6 | Knull revealed |
| kib1-5 | King in Black (2020-21) #1-5 | Knull's invasion |

### The Anti-Monitor
| ID | Comic | Use |
|---|---|---|
| coie1-12 | Crisis on Infinite Earths #1-12 (1985-86) | The whole story |

### Darkseid
| ID | Comic | Use |
|---|---|---|
| fp1 | Forever People #1 (1971) | First full appearance |
| fc1-7 | Final Crisis #1-7 (2008) | His "death" |

### Thanos vs Galactus
| ID | Comic | Use |
|---|---|---|
| ig1-6 | The Infinity Gauntlet #1-6 (1991) | Thanos at full power, cosmic beings vs Thanos |
| ff48-50 | Fantastic Four #48-50 (1966) | Galactus at his peak |

### The Celestials
| ID | Comic | Use |
|---|---|---|
| eternals1 | The Eternals #1 (1976) | First appearance |
| — | Thor's Celestials saga (late 1970s-1980) — **verify** issues | Judgment of Earth |

### The Speed Force
| ID | Comic | Use |
|---|---|---|
| flash91 | The Flash vol. 2 #91 (1994) — **verify** | Speed Force introduced |
| flash95-100 | The Flash vol. 2 #95-100 (1995, "Terminal Velocity") | The Speed Force explained |

### 5 cosmic beings stronger than Galactus
| ID | Comic | Being |
|---|---|---|
| st138 | Strange Tales #138 (1965) | Eternity |
| st157 | Strange Tales #157 (1967) | Living Tribunal |
| ff511 | Fantastic Four #511 (2004) | The One-Above-All |
| ig1-6 | The Infinity Gauntlet #1-6 (1991) | Cosmic hierarchy vs Thanos |
| eternals1 | The Eternals #1 (1976) | Celestials |

### VID900 · Fantomah (public domain, nothing to buy)
Jungle Comics #16 (1941), already linked from Wikimedia Commons in the Sources tab.

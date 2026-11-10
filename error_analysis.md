# Error analysis: bằng chứng test thật

W8 mục 24 yêu cầu sinh viên tự error analysis. Bảng dưới lưu 3 false positives, 3 false negatives và 2 ca sai tự tin nhất; có thể trùng index giữa các nhóm. Các dấu hiệu tự động và hypothesis là gợi ý kiểm chứng, không phải kết luận nguyên nhân. Sinh viên cần tự bổ sung phân loại và giải thích.

| index | group | true | pred | confidence | error type (candidate) | hypothesis (unverified) |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | false_positive | 0 | 1 | 0.9338 | negation cue, contrast cue, truncation | Kiểm tra scope phủ định và sentiment trong đoạn được giữ ; Kiểm tra mệnh đề tương phản và nhãn tổng thể ; Sentiment quan trọng có thể ở phần bị cắt; so sánh T256 |
| 6 | false_positive | 0 | 1 | 0.9766 | negation cue, contrast cue, truncation | Kiểm tra scope phủ định và sentiment trong đoạn được giữ ; Kiểm tra mệnh đề tương phản và nhãn tổng thể ; Sentiment quan trọng có thể ở phần bị cắt; so sánh T256 |
| 17 | false_positive | 0 | 1 | 0.5756 | negation cue, truncation | Kiểm tra scope phủ định và sentiment trong đoạn được giữ ; Sentiment quan trọng có thể ở phần bị cắt; so sánh T256 |
| 12512 | false_negative | 1 | 0 | 0.6673 | negation cue, contrast cue, truncation | Kiểm tra scope phủ định và sentiment trong đoạn được giữ ; Kiểm tra mệnh đề tương phản và nhãn tổng thể ; Sentiment quan trọng có thể ở phần bị cắt; so sánh T256 |
| 12518 | false_negative | 1 | 0 | 0.5443 | context / label ambiguity cần xem thủ công | Đối chiếu ngữ cảnh giữ lại, sarcasm và tính rõ ràng của nhãn trước khi gán nguyên nhân |
| 12525 | false_negative | 1 | 0 | 0.9447 | negation cue, contrast cue, truncation | Kiểm tra scope phủ định và sentiment trong đoạn được giữ ; Kiểm tra mệnh đề tương phản và nhãn tổng thể ; Sentiment quan trọng có thể ở phần bị cắt; so sánh T256 |
| 4118 | most_confident_wrong | 0 | 1 | 0.9965 | negation cue | Kiểm tra scope phủ định và sentiment trong đoạn được giữ |
| 19650 | most_confident_wrong | 1 | 0 | 0.9944 | negation cue, contrast cue, truncation | Kiểm tra scope phủ định và sentiment trong đoạn được giữ ; Kiểm tra mệnh đề tương phản và nhãn tổng thể ; Sentiment quan trọng có thể ở phần bị cắt; so sánh T256 |

## Test #4 - false_positive

True=0, Pred=1, confidence=0.933758, P(positive)=0.933758.

Tokens=163, giữ=128, truncated=True, UNK=0.

Text gốc:

> First off let me say, If you haven't enjoyed a Van Damme movie since bloodsport, you probably will not like this movie. Most of these movies may not have the best plots or best actors but I enjoy these kinds of movies for what they are. This movie is much better than any of the movies the other action guys (Segal and Dolph) have thought about putting out the past few years. Van Damme is good in the movie, the movie is only worth watching to Van Damme fans. It is not as good as Wake of Death (which i highly recommend to anyone of likes Van Damme) or In hell but, in my opinion it's worth watching. It has the same type of feel to it as Nowhere to Run. Good fun stuff!

Token input thực tế:

```text
[CLS] first off let me say , if you haven ' t enjoyed a van dam ##me movie since blood ##sport , you probably will not like this movie . most of these movies may not have the best plots or best actors but i enjoy these kinds of movies for what they are . this movie is much better than any of the movies the other action guys ( sega ##l and do ##lp ##h ) have thought about putting out the past few years . van dam ##me is good in the movie , the movie is only worth watching to van dam ##me fans . it is not as good as wake of death ( which i highly recommend to anyone of likes van [SEP]
```

Candidate types: negation cue, contrast cue, truncation.

Hypothesis chưa kiểm chứng: Kiểm tra scope phủ định và sentiment trong đoạn được giữ; Kiểm tra mệnh đề tương phản và nhãn tổng thể; Sentiment quan trọng có thể ở phần bị cắt; so sánh T256.

**Sinh viên tự phân tích:** kiểm tra negation/sarcasm/long-range/ambiguous sentiment/domain expression/rare word/tokenization/context/label ambiguity; trích evidence cụ thể, nêu cách kiểm tra giả thuyết. Confidence cao có giải thích được vì sao sai không?

## Test #6 - false_positive

True=0, Pred=1, confidence=0.976569, P(positive)=0.976569.

Tokens=376, giữ=128, truncated=True, UNK=0.

Text gốc:

> Isaac Florentine has made some of the best western Martial Arts action movies ever produced. In particular US Seals 2, Cold Harvest, Special Forces and Undisputed 2 are all action classics. You can tell Isaac has a real passion for the genre and his films are always eventful, creative and sharp affairs, with some of the best fight sequences an action fan could hope for. In particular he has found a muse with Scott Adkins, as talented an actor and action performer as you could hope for. This is borne out with Special Forces and Undisputed 2, but unfortunately The Shepherd just doesn't live up to their abilities.<br /><br />There is no doubt that JCVD looks better here fight-wise than he has done in years, especially in the fight he has (for pretty much no reason) in a prison cell, and in the final showdown with Scott, but look in his eyes. JCVD seems to be dead inside. There's nothing in his eyes at all. It's like he just doesn't care about anything throughout the whole film. And this is the leading man.<br /><br />There are other dodgy aspects to the film, script-wise and visually, but the main problem is that you are utterly unable to empathise with the hero of the film. A genuine shame as I know we all wanted this film to be as special as it genuinely could have been. There are some good bits, mostly the action scenes themselves. This film had a terrific director and action choreographer, and an awesome opponent for JCVD to face down. This could have been the one to bring the veteran action star back up to scratch in the balls-out action movie stakes.<br /><br />Sincerely a shame that this didn't happen.

Token input thực tế:

```text
[CLS] isaac fl ##ore ##ntine has made some of the best western martial arts action movies ever produced . in particular us seals 2 , cold harvest , special forces and und ##is ##puted 2 are all action classics . you can tell isaac has a real passion for the genre and his films are always event ##ful , creative and sharp affairs , with some of the best fight sequences an action fan could hope for . in particular he has found a muse with scott ad ##kins , as talented an actor and action performer as you could hope for . this is borne out with special forces and und ##is ##puted 2 , but unfortunately the shepherd just doesn ' t live up to [SEP]
```

Candidate types: negation cue, contrast cue, truncation.

Hypothesis chưa kiểm chứng: Kiểm tra scope phủ định và sentiment trong đoạn được giữ; Kiểm tra mệnh đề tương phản và nhãn tổng thể; Sentiment quan trọng có thể ở phần bị cắt; so sánh T256.

**Sinh viên tự phân tích:** kiểm tra negation/sarcasm/long-range/ambiguous sentiment/domain expression/rare word/tokenization/context/label ambiguity; trích evidence cụ thể, nêu cách kiểm tra giả thuyết. Confidence cao có giải thích được vì sao sai không?

## Test #17 - false_positive

True=0, Pred=1, confidence=0.575591, P(positive)=0.575591.

Tokens=186, giữ=128, truncated=True, UNK=0.

Text gốc:

> I just finished watching this movie and am disappointed to say that I didn't enjoy it a bit. It is so slow Slow and uninteresting. This kid from Harry Potter plays a shy teenager with an rude mother, and then one day the rude mother tells the kid to find a job so that they could accommodate an old guy apparently having no place to live has started to live with his family and therefore the kid goes to work for a old lady. And this old lady who is living all alone teaches him about girls, driving car and life! I couldn't get how an 18 year old guy enjoy spending time with an awful lady in her 80s. Sorry if my comments on this movie has bothered people who might have enjoyed it, I could be wrong as I am not British and may not understand the social and their family structure and way of life. Mostly the movie is made for the British audience.

Token input thực tế:

```text
[CLS] i just finished watching this movie and am disappointed to say that i didn ' t enjoy it a bit . it is so slow slow and un ##int ##eres ##ting . this kid from harry potter plays a shy teenager with an rude mother , and then one day the rude mother tells the kid to find a job so that they could accommodate an old guy apparently having no place to live has started to live with his family and therefore the kid goes to work for a old lady . and this old lady who is living all alone teaches him about girls , driving car and life ! i couldn ' t get how an 18 year old guy enjoy spending time [SEP]
```

Candidate types: negation cue, truncation.

Hypothesis chưa kiểm chứng: Kiểm tra scope phủ định và sentiment trong đoạn được giữ; Sentiment quan trọng có thể ở phần bị cắt; so sánh T256.

**Sinh viên tự phân tích:** kiểm tra negation/sarcasm/long-range/ambiguous sentiment/domain expression/rare word/tokenization/context/label ambiguity; trích evidence cụ thể, nêu cách kiểm tra giả thuyết. Confidence cao có giải thích được vì sao sai không?

## Test #12512 - false_negative

True=1, Pred=0, confidence=0.667338, P(positive)=0.332662.

Tokens=308, giữ=128, truncated=True, UNK=0.

Text gốc:

> Its a very sensitive portrayal of life with unquenched or constrained desires. What does one do with desire in a culture and society with rigid norms? One husband finds outlet with the immigrant - since immigrants don't belong or aren't accepted, they don't need to conform and dam their desires. The other husband looks for solace in spirituality and tries to evaporate his desire into nothingness. It fails - of course - and he breaks down in the last scene for multiple reasons. Sita still cared enough for him to find that moment to let him know that he is not responsible for her deviant outlet to her blocked desires. The mother in her still couldn't find the strength to destroy his myth. She sees him as a child who is glorifying himself in his lust-control but should she give him the opportunity to finally grow up? Both the wives find courage and togetherness through their shared rejection by their husband. <br /><br />But the final act of rejection was by the grandmother - she could not break free from her rusted mindset to accept Sita's desire. A decade and more of receiving care was not enough to break the shackles of her culture.<br /><br />Seems like it was easiest for the househelp to let his desires flow - since he's anyway damned by his culture - being at the bottom of the hierarchy. Since there is anyway no respect and expectations, might as well taste sin.

Token input thực tế:

```text
[CLS] its a very sensitive portrayal of life with un ##que ##nched or constrained desires . what does one do with desire in a culture and society with rigid norms ? one husband finds outlet with the immigrant - since immigrants don ' t belong or aren ' t accepted , they don ' t need to conform and dam their desires . the other husband looks for sol ##ace in spirituality and tries to eva ##por ##ate his desire into nothing ##ness . it fails - of course - and he breaks down in the last scene for multiple reasons . sit ##a still cared enough for him to find that moment to let him know that he is not responsible for her devi ##ant outlet [SEP]
```

Candidate types: negation cue, contrast cue, truncation.

Hypothesis chưa kiểm chứng: Kiểm tra scope phủ định và sentiment trong đoạn được giữ; Kiểm tra mệnh đề tương phản và nhãn tổng thể; Sentiment quan trọng có thể ở phần bị cắt; so sánh T256.

**Sinh viên tự phân tích:** kiểm tra negation/sarcasm/long-range/ambiguous sentiment/domain expression/rare word/tokenization/context/label ambiguity; trích evidence cụ thể, nêu cách kiểm tra giả thuyết. Confidence cao có giải thích được vì sao sai không?

## Test #12518 - false_negative

True=1, Pred=0, confidence=0.544277, P(positive)=0.455723.

Tokens=62, giữ=62, truncated=False, UNK=0.

Text gốc:

> The theme is controversial and the depiction of the hypocritical and sexually starved india is excellent.Nothing more to this film.There is a lack of good dialogues(why was the movie in english??). There was lack of continuity and lack of passion/emotion in the acting.

Token input thực tế:

```text
[CLS] the theme is controversial and the depiction of the h ##yp ##oc ##rit ##ical and sexually starved india is excellent . nothing more to this film . there is a lack of good dialogues ( why was the movie in english ? ? ) . there was lack of continuity and lack of passion / emotion in the acting . [SEP]
```

Candidate types: context / label ambiguity cần xem thủ công.

Hypothesis chưa kiểm chứng: Đối chiếu ngữ cảnh giữ lại, sarcasm và tính rõ ràng của nhãn trước khi gán nguyên nhân.

**Sinh viên tự phân tích:** kiểm tra negation/sarcasm/long-range/ambiguous sentiment/domain expression/rare word/tokenization/context/label ambiguity; trích evidence cụ thể, nêu cách kiểm tra giả thuyết. Confidence cao có giải thích được vì sao sai không?

## Test #12525 - false_negative

True=1, Pred=0, confidence=0.944682, P(positive)=0.055318.

Tokens=223, giữ=128, truncated=True, UNK=0.

Text gốc:

> Although Bullet In The Brain is, without question, superior amongst short films, it largely seems more like a short piece of writing than a film. And it is a little hard to feel too sorry for the teacher when his smart ass remarks get him shot. But after the bullet enters his brain we begin to understand a little bit about why he became so jaded with life in the first place. There is an awful amount of detail packed into this reasonably short film and this is what makes me feel that it should have been extended a little bit - it seems like there's almost too much to take in at once as the details come flying at you so fast. A slightly more relaxed pace and a less po-faced narrator in the final section would have benefitted this film a little bit. Despite these complaints, there is no denying that Bullet In The Brain is a quite stupendous work compared to many short, and even full length films. The makers should be applauded for trying to make such a basically emotional and literate film in the current climate of quick jokes and Hollywood action.

Token input thực tế:

```text
[CLS] although bullet in the brain is , without question , superior amongst short films , it largely seems more like a short piece of writing than a film . and it is a little hard to feel too sorry for the teacher when his smart ass remarks get him shot . but after the bullet enters his brain we begin to understand a little bit about why he became so jade ##d with life in the first place . there is an awful amount of detail packed into this reasonably short film and this is what makes me feel that it should have been extended a little bit - it seems like there ' s almost too much to take in at once as the details [SEP]
```

Candidate types: negation cue, contrast cue, truncation.

Hypothesis chưa kiểm chứng: Kiểm tra scope phủ định và sentiment trong đoạn được giữ; Kiểm tra mệnh đề tương phản và nhãn tổng thể; Sentiment quan trọng có thể ở phần bị cắt; so sánh T256.

**Sinh viên tự phân tích:** kiểm tra negation/sarcasm/long-range/ambiguous sentiment/domain expression/rare word/tokenization/context/label ambiguity; trích evidence cụ thể, nêu cách kiểm tra giả thuyết. Confidence cao có giải thích được vì sao sai không?

## Test #4118 - most_confident_wrong

True=0, Pred=1, confidence=0.996477, P(positive)=0.996477.

Tokens=73, giữ=73, truncated=False, UNK=0.

Text gốc:

> This is definitely one of the best Kung fu movies in the history of Cinema. The screenplay is really well done (which is not often the case for this type of movies) and you can see that Chuck (in one of his first role)is a great actor. The final fight with the sherif deputy in the bullring is a masterpiece!

Token input thực tế:

```text
[CLS] this is definitely one of the best kung fu movies in the history of cinema . the screenplay is really well done ( which is not often the case for this type of movies ) and you can see that chuck ( in one of his first role ) is a great actor . the final fight with the she ##ri ##f deputy in the bull ##ring is a masterpiece ! [SEP]
```

Candidate types: negation cue.

Hypothesis chưa kiểm chứng: Kiểm tra scope phủ định và sentiment trong đoạn được giữ.

**Sinh viên tự phân tích:** kiểm tra negation/sarcasm/long-range/ambiguous sentiment/domain expression/rare word/tokenization/context/label ambiguity; trích evidence cụ thể, nêu cách kiểm tra giả thuyết. Confidence cao có giải thích được vì sao sai không?

## Test #19650 - most_confident_wrong

True=1, Pred=0, confidence=0.994434, P(positive)=0.005566.

Tokens=662, giữ=128, truncated=True, UNK=0.

Text gốc:

> I just saw this for the first time in 10 or 15 years...maybe close to 20. In some ways, it was better than I remembered...in other, it was MUCH worse.<br /><br />First of all, there's the music. It's just plain awful. There are only 5 songs in the movie, most of them used more than once. The opening song is shrieked by a chorus of annoying children, and the disco-y title track is performed by Rick Dees. It doesn't get any worse than that. Even the background music is terrible, with much of it repeating the themes of the other nauseating tunes. We also get some truly lame slapstick, mostly in the opening credits.<br /><br />On the other hand, Bill Murray is spot on brilliant as usual...you have to wonder if he ad-libbed the whole thing, or if the writers just gave him all the funny lines. Or maybe he's just that great- turning a weak script into comic genius. The best part are his surreal PA announcements. ("Lobsters...get out of here...you're a menace!")<br /><br />You also get a lot more character development than you have any right to expect in a movie like this. At least half the characters seem like real people...and mostly real people you would like to have around. Even "Spaz" gets to do a more than any other Eddie Deezen-type character ever did, and when he gets the girl, it's plausible. (She's not absurdly hot, but he doesn't automatically pair up with one of the nerd girls- see "Revenge Of The Nerds" for examples of both cinematic phenomenons.)<br /><br />And when the plot seems clichéd...well, ya gotta wonder if it wasn't a cliché yet when they made this. While it wasn't the first summer camp movie- ya gotta go back at least to "The Parent Trap"- it's certainly the movie that made it it's own genre. In fact, I was surprised that there was no Talent Show scene..."Wet Hot American Summer" spoofed the summer-camp genre so perfectly, I just assumed everything in it came straight out of Meatballs. (I also half-expected Jon Cryer to pull up in a convertible with a chimp, thanks to "Mr. Show's" epic camp-olympiad spoof "Monk Academy") <br /><br />Anyway, this one seems to be vanishing a little as far as the late-70's/early 80's comedies- it's not a cable staple anymore, and certainly doesn't have the cult following of Caddyshack or Animal House. I was pleased to catch it on Showtime today- and in High Definition at that! Sure, it's pretty awful in spots, but you could do a lot worse in a 70's/80's teen comedy. And again, Murray is a genius.

Token input thực tế:

```text
[CLS] i just saw this for the first time in 10 or 15 years . . . maybe close to 20 . in some ways , it was better than i remembered . . . in other , it was much worse . < br / > < br / > first of all , there ' s the music . it ' s just plain awful . there are only 5 songs in the movie , most of them used more than once . the opening song is shrieked by a chorus of annoying children , and the disco - y title track is performed by rick dee ##s . it doesn ' t get any worse than that . even the background music is terrible [SEP]
```

Candidate types: negation cue, contrast cue, truncation.

Hypothesis chưa kiểm chứng: Kiểm tra scope phủ định và sentiment trong đoạn được giữ; Kiểm tra mệnh đề tương phản và nhãn tổng thể; Sentiment quan trọng có thể ở phần bị cắt; so sánh T256.

**Sinh viên tự phân tích:** kiểm tra negation/sarcasm/long-range/ambiguous sentiment/domain expression/rare word/tokenization/context/label ambiguity; trích evidence cụ thể, nêu cách kiểm tra giả thuyết. Confidence cao có giải thích được vì sao sai không?

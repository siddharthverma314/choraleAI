import matplotlib.pyplot as plt

scalars = [(0,1.5679088830947876,-0.5562707781791687),
           (1,0.6278689503669739,-0.31358030438423157),
           (2,0.6274133920669556,-0.3134981691837311),
           (3,0.6271280646324158,-0.3133755028247833),
           (4,0.6270679235458374,-0.31340283155441284),
           (5,0.627058744430542,-0.3133794069290161),
           (6,0.6272017359733582,-0.31344062089920044),
           (7,0.6270792484283447,-0.31338411569595337),
           (8,0.6271673440933228,-0.31340786814689636),
           (9,0.6271914839744568,-0.31342703104019165),
           (10,0.6273049712181091,-0.31349191069602966),
           (11,0.6271330714225769,-0.31335803866386414),
           (12,0.6271217465400696,-0.3133781850337982),
           (13,0.6271716952323914,-0.3134257197380066),
           (14,0.6271652579307556,-0.31344136595726013),
           (15,0.6271267533302307,-0.3133968710899353),
           (16,0.6272067427635193,-0.3134419620037079),
           (17,0.6271634697914124,-0.3133980929851532),
           (18,0.6272810101509094,-0.313427209854126),
           (19,0.6271454095840454,-0.3134050965309143),
           (20,0.6272287368774414,-0.3133939206600189),
           (21,0.6270764470100403,-0.3134075701236725),
           (22,0.627084493637085,-0.31341060996055603),
           (23,0.6274049878120422,-0.31346258521080017),
           (24,0.6272267699241638,-0.3134956657886505),
           (25,0.6273844838142395,-0.31348153948783875),
           (26,0.6273863315582275,-0.3135487139225006),
           (27,0.6270954012870789,-0.31340867280960083),
           (28,0.6272236704826355,-0.31344330310821533),
           (29,0.6272460222244263,-0.3133959472179413),
           (30,0.6271787285804749,-0.3133891522884369),
           (31,0.6271795034408569,-0.31343206763267517),
           (32,0.6270617842674255,-0.3134051263332367),
           (33,0.6271820068359375,-0.3134392499923706),
           (34,0.6272656917572021,-0.31345632672309875),
           (35,0.6272042393684387,-0.3134543001651764),
           (36,0.6271316409111023,-0.313402384519577),
           (37,0.6271588206291199,-0.3134358823299408),
           (38,0.6272274255752563,-0.3134878873825073),
           (39,0.6271159052848816,-0.31340375542640686),
           (40,0.6271230578422546,-0.31340500712394714),
           (41,0.6271313428878784,-0.31338435411453247),
           (42,0.6270979642868042,-0.31343764066696167),
           (43,0.6273244619369507,-0.3134387135505676),
           (44,0.6272637248039246,-0.31343239545822144),
           (45,0.6271771788597107,-0.3134201169013977),
           (46,0.6272635459899902,-0.313403457403183),
           (47,0.6273069381713867,-0.31348443031311035),
           (48,0.6271530985832214,-0.3134204149246216),
           (49,0.6271510124206543,-0.3133977949619293),
           (50,0.6271366477012634,-0.31338977813720703),
           (51,0.6272200345993042,-0.31342604756355286),
           (52,0.6272410750389099,-0.3134393095970154),
           (53,0.627249538898468,-0.3134321868419647),
           (54,0.6272233724594116,-0.31343892216682434),
           (55,0.6272887587547302,-0.3134022057056427),
           (56,0.6271770596504211,-0.3134251832962036),
           (57,0.6271462440490723,-0.3134117126464844),
           (58,0.6271973848342896,-0.3133995532989502),
           (59,0.6272385716438293,-0.3134428560733795),
           (60,0.6272443532943726,-0.31342506408691406),
           (61,0.6271337866783142,-0.3134061098098755),
           (62,0.6271888613700867,-0.31342813372612),
           (63,0.6272960901260376,-0.31342971324920654),
           (64,0.6271049976348877,-0.31341370940208435),
           (65,0.6271629333496094,-0.31343600153923035),
           (66,0.627187192440033,-0.31342846155166626),
           (67,0.6272647976875305,-0.3134041726589203),
           (68,0.6272234320640564,-0.31341513991355896),
           (69,0.6272016167640686,-0.3134058713912964),
           (70,0.6271688342094421,-0.3134635090827942),
           (71,0.6272282600402832,-0.31344419717788696),
           (72,0.6271711587905884,-0.31342729926109314),
           (73,0.6271443963050842,-0.3134349584579468),
           (74,0.6272470355033875,-0.31341925263404846),
           (75,0.6271275877952576,-0.3134286105632782),
           (76,0.6271594166755676,-0.3134267032146454),
           (77,0.6272326111793518,-0.3134402930736542),
           (78,0.627156674861908,-0.31343212723731995),
           (79,0.6272069215774536,-0.3134048581123352),
           (80,0.6272351145744324,-0.3134268820285797),
           (81,0.6271964907646179,-0.31343165040016174),
           (82,0.6270785927772522,-0.3134404122829437),
           (83,0.6270930171012878,-0.31339752674102783),
           (84,0.6273279190063477,-0.31345221400260925),
           (85,0.62724369764328,-0.3134179711341858),
           (86,0.6271683573722839,-0.3134443461894989),
           (87,0.6271187663078308,-0.31342336535453796),
           (88,0.6271739602088928,-0.3134431540966034),
           (89,0.6271743178367615,-0.31342509388923645),
           (90,0.6271891593933105,-0.3134320080280304),
           (91,0.6271258592605591,-0.31341251730918884),
           (92,0.6272526979446411,-0.3134682774543762),
           (93,0.6270912885665894,-0.31342369318008423),
           (94,0.6271328926086426,-0.31341442465782166),
           (95,0.6271728277206421,-0.31343016028404236),
           (96,0.6271791458129883,-0.3134331703186035),
           (97,0.6271315217018127,-0.31342166662216187),
           (98,0.6271359324455261,-0.31340083479881287),
           (99,0.6272006034851074,-0.31344074010849),
           (100,0.6270975470542908,-0.3134097456932068),
           (101,0.6272467374801636,-0.3134411871433258),
           (102,0.6272672414779663,-0.3134187161922455),
           (103,0.627300500869751,-0.3134302794933319),
           (104,0.6272298693656921,-0.3134218156337738),
           (105,0.6272282004356384,-0.3134203851222992),
           (106,0.6271534562110901,-0.3134138584136963),
           (107,0.627219557762146,-0.31341812014579773),
           (108,0.6272075772285461,-0.3134632408618927),
           (109,0.6271953582763672,-0.3134056329727173),
           (110,0.6271506547927856,-0.31341999769210815),
           (111,0.6270804405212402,-0.3134293854236603),
           (112,0.6271948218345642,-0.3134247362613678)]

x = [i[0] for i in scalars]
disc = [i[1] for i in scalars]
gen = [i[2]*-1 for i in scalars]

plt.plot(x, disc)
plt.plot(x, gen)
plt.show()

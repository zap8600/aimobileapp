package io.github.zap8600.aimobileapp.tokenization.CodegenTokenizer

internal val byteEncoder: Map<Int, String> by lazy {
    hashMapOf<Int, String>().apply {
        put(33, "!")
        put(34, "\"")
        put(35, "#")
        put(36, "$")
        put(37, "%")
        put(38, "&")
        put(39, "'")
        put(40, "(")
        put(41, ")")
        put(42, "*")
        put(43, "+")
        put(44, ",")
        put(45, "-")
        put(46, ".")
        put(47, "/")
        put(48, "0")
        put(49, "1")
        put(50, "2")
        put(51, "3")
        put(52, "4")
        put(53, "5")
        put(54, "6")
        put(55, "7")
        put(56, "8")
        put(57, "9")
        put(58, ":")
        put(59, ";")
        put(60, "<")
        put(61, "=")
        put(62, ">")
        put(63, "?")
        put(64, "@")
        put(65, "A")
        put(66, "B")
        put(67, "C")
        put(68, "D")
        put(69, "E")
        put(70, "F")
        put(71, "G")
        put(72, "H")
        put(73, "I")
        put(74, "J")
        put(75, "K")
        put(76, "L")
        put(77, "M")
        put(78, "N")
        put(79, "O")
        put(80, "P")
        put(81, "Q")
        put(82, "R")
        put(83, "S")
        put(84, "T")
        put(85, "U")
        put(86, "V")
        put(87, "W")
        put(88, "X")
        put(89, "Y")
        put(90, "Z")
        put(91, "[")
        put(92, "\\")
        put(93, "]")
        put(94, "^")
        put(95, "_")
        put(96, "`")
        put(97, "a")
        put(98, "b")
        put(99, "c")
        put(100, "d")
        put(101, "e")
        put(102, "f")
        put(103, "g")
        put(104, "h")
        put(105, "i")
        put(106, "j")
        put(107, "k")
        put(108, "l")
        put(109, "m")
        put(110, "n")
        put(111, "o")
        put(112, "p")
        put(113, "q")
        put(114, "r")
        put(115, "s")
        put(116, "t")
        put(117, "u")
        put(118, "v")
        put(119, "w")
        put(120, "x")
        put(121, "y")
        put(122, "z")
        put(123, "{")
        put(124, "|")
        put(125, "}")
        put(126, "~")
        put(161, "¡")
        put(162, "¢")
        put(163, "£")
        put(164, "¤")
        put(165, "¥")
        put(166, "¦")
        put(167, "§")
        put(168, "¨")
        put(169, "©")
        put(170, "ª")
        put(171, "«")
        put(172, "¬")
        put(174, "®")
        put(175, "¯")
        put(176, "°")
        put(177, "±")
        put(178, "²")
        put(179, "³")
        put(180, "´")
        put(181, "µ")
        put(182, "¶")
        put(183, "·")
        put(184, "¸")
        put(185, "¹")
        put(186, "º")
        put(187, "»")
        put(188, "¼")
        put(189, "½")
        put(190, "¾")
        put(191, "¿")
        put(192, "À")
        put(193, "Á")
        put(194, "Â")
        put(195, "Ã")
        put(196, "Ä")
        put(197, "Æ")
        put(198, "Ç")
        put(199, "È")
        put(200, "É")
        put(201, "Ê")
        put(202, "Ë")
        put(203, "Ì")
        put(204, "Í")
        put(205, "Î")
        put(206, "Ï")
        put(207, "Ð")
        put(208, "Ñ")
        put(209, "Ò")
        put(210, "Ó")
        put(211, "Ô")
        put(212, "Õ")
        put(213, "Ö")
        put(214, "×")
        put(215, "Ø")
        put(216, "Ù")
        put(217, "Ú")
        put(218, "Û")
        put(219, "Ü")
        put(220, "Ý")
        put(221, "Þ")
        put(222, "ß")
        put(223, "à")
        put(224, "á")
        put(225, "â")
        put(226, "ã")
        put(227, "ä")
        put(228, "å")
        put(229, "æ")
        put(230, "ç")
        put(231, "è")
        put(232, "é")
        put(233, "ê")
        put(234, "ê")
        put(235, "ë")
        put(236, "ì")
        put(237, "í")
        put(238, "î")
        put(239, "ï")
        put(240, "ð")
        put(241, "ñ")
        put(242, "ò")
        put(243, "ó")
        put(244, "ô")
        put(245, "õ")
        put(246, "ö")
        put(247, "÷")
        put(248, "ø")
        put(249, "ù")
        put(250, "ú")
        put(251, "û")
        put(252, "ü")
        put(253, "ý")
        put(254, "þ")
        put(255, "ÿ")
        put(0, "Ā")
        put(1, "ā")
        put(2, "Ă")
        put(3, "ă")
        put(4, "Ą")
        put(5, "ą")
        put(6, "Ć")
        put(7, "ć")
        put(8, "Ĉ")
        put(9, "ĉ")
        put(10, "Ċ")
        put(11, "ċ")
        put(12, "Č")
        put(13, "č")
        put(14, "Ď")
        put(15, "ď")
        put(16, "Đ")
        put(17, "đ")
        put(18, "Ē")
        put(19, "ē")
        put(20, "Ĕ")
        put(21, "ĕ")
        put(22, "Ė")
        put(23, "ė")
        put(24, "Ę")
        put(25, "ę")
        put(26, "Ě")
        put(27, "ě")
        put(28, "Ĝ")
        put(29, "ĝ")
        put(30, "Ğ")
        put(31, "ğ")
        put(32, "Ġ")
        put(127, "ġ")
        put(128, "Ģ")
        put(129, "ģ")
        put(130, "Ĥ")
        put(131, "ĥ")
        put(132, "Ħ")
        put(133, "ħ")
        put(134, "Ĩ")
        put(135, "ĩ")
        put(136, "Ī")
        put(137, "ī")
        put(138, "Ĭ")
        put(139, "ĭ")
        put(140, "Į")
        put(141, "į")
        put(142, "İ")
        put(143, "ı")
        put(144, "Ĳ")
        put(145, "ĳ")
        put(146, "Ĵ")
        put(147, "ĵ")
        put(148, "Ķ")
        put(149, "ķ")
        put(150, "ĸ")
        put(151, "Ĺ")
        put(152, "ĺ")
        put(153, "Ļ")
        put(154, "ļ")
        put(155, "Ľ")
        put(156, "ľ")
        put(157, "Ŀ")
        put(158, "ŀ")
        put(159, "Ł")
        put(160, "ł")
        put(173, "Ń")
    }
}

internal val byteDecoder by lazy {
    byteEncoder.entries.associateBy({ it.value }) { it.key }
}
/* Game Prompt Studio — builds STAR (Situation, Task, Action, Result) prompts
   for Gemini Canvas from the form on the left. No dependencies.
   Bilingual: every text is an { en, id } pair. The interface language and the
   prompt language are chosen separately. */

// ---------- Interface text ----------

const UI = {
  en: {
    tagline: 'Design Islamic educational games, get a STAR prompt, paste it into Gemini Canvas.',
    presetsAria: 'Presets',
    uiLangAria: 'Interface language',
    startFrom: 'Start from:',
    presetSahabat: '🌙 Sahabat Quiz',
    presetArabic: '📖 Arabic Vocab',
    presetHero: '⚔️ Muslim Heroes',
    presetHijaiyah: '🔤 Hijaiyah Match',
    presetRukunIman: '☪️ Rukun Iman',
    presetRukunIslam: '🕋 Rukun Islam',
    presetTebakAyat: '📜 Guess the Ayah',
    presetQuran: '🕌 Quran Explorer',
    presetSholat: '🤲 Prayer Recitations',
    presetDoa: '🕊️ Daily Du\'a',
    s1: 'Game basics',
    fTitle: 'Game title',
    phTitle: 'e.g. Sahabat Rasul Challenge',
    fGenre: 'Game type',
    fTheme: 'Theme / topic',
    phTheme: 'e.g. companions of the Prophet in Medina',
    fConcept: 'Core idea',
    optional: '(optional)',
    phConcept: 'Describe your idea in 1–3 sentences: what does the player do and why is it fun?',
    fAudience: 'Target players',
    fPurpose: 'Purpose',
    fObjective: 'Learning objective',
    phObjective: 'e.g. Students can name 10 Sahabat and their key contributions to Islam',
    s2: 'Gameplay',
    fLevels: 'Number of levels',
    fLives: 'Starting lives',
    fSession: 'Session length',
    fDifficulty: 'Default difficulty',
    easy: 'Easy', normal: 'Normal', hard: 'Hard',
    s3: 'Features',
    s4: 'Look & feel',
    fStyle: 'Visual style',
    fPalette: 'Color palette',
    cPrimary: 'Primary', cAccent: 'Accent', cBg: 'Background',
    fGameLang: 'In-game language',
    fMood: 'Mood',
    s5: 'Prompt & code output',
    fPromptLang: 'Prompt language',
    promptLangHint: 'The language of the prompt sent to Gemini. It can differ from the language used inside the game.',
    fmtSingle: 'Single HTML file',
    fmtSingleDesc: 'HTML, CSS and JS in one <code>index.html</code>. Best for previewing in Gemini Canvas.',
    fmtMulti: 'Separate files',
    fmtMultiDesc: '<code>index.html</code>, <code>style.css</code> and <code>script.js</code>, structured like a project folder for localhost.',
    optComments: 'Well-commented code',
    optLocalhost: 'Include localhost run instructions',
    optNext: 'Suggest 3 next improvements',
    fExtra: 'Extra requirements',
    phExtra: "Anything else, e.g. 'Show our madrasah name on the title screen', 'Add Bismillah before the game starts', 'Include Arabic calligraphy decorations'",
    outAria: 'Generated prompt',
    outTitle: 'Your STAR prompt',
    legend: ['S · Situation', 'T · Task', 'A · Action', 'R · Result'],
    copy: '📋 Copy prompt',
    download: '⬇️ Download .txt',
    openGemini: '✨ Open Gemini',
    howtoTitle: 'How to use it in Gemini Canvas',
    howtoSteps: `<li>Click <strong>Copy prompt</strong>, then <strong>Open Gemini</strong>.</li>
      <li>In the Gemini prompt bar, turn on <strong>Canvas</strong>.</li>
      <li>Paste the prompt and send it. Gemini writes the code and shows a live preview.</li>
      <li>Play the game in the preview. To make changes, ask in the chat (see the follow-up prompts below).</li>
      <li>To run it on localhost: save the code as <code>index.html</code> in a folder, run <code>python3 -m http.server 8000</code> in that folder, then open <code>http://localhost:8000</code>.</li>`,
    followTitle: 'Follow-up prompts to keep improving',
    followHint: 'Click to copy',
    words: 'words', chars: 'characters',
    toastCopied: 'Prompt copied. Paste it into Gemini Canvas.',
    toastFollow: 'Follow-up prompt copied',
    toastPreset: (t) => `Loaded "${t}"`,
    fContext: 'Knowledge context',
    dropText: 'Click or drag files here',
    contextLoaded: '✅ Context loaded',
    contextClear: 'Clear',
    contextUse: 'Include uploaded context in prompt',
    contextSection: 'Reference material (from uploaded documents)',
    contextParsing: 'Reading files...',
  },
  id: {
    tagline: 'Rancang game edukasi Islam, dapatkan prompt STAR, lalu tempel ke Gemini Canvas.',
    presetsAria: 'Contoh siap pakai',
    uiLangAria: 'Bahasa tampilan',
    startFrom: 'Mulai dari:',
    presetSahabat: '🌙 Kuis Sahabat',
    presetArabic: '📖 Kosa Kata Arab',
    presetHero: '⚔️ Pahlawan Muslim',
    presetHijaiyah: '🔤 Cocokkan Hijaiyah',
    presetRukunIman: '☪️ Rukun Iman',
    presetRukunIslam: '🕋 Rukun Islam',
    presetTebakAyat: '📜 Tebak Ayat',
    presetQuran: '🕌 Jelajah Al-Quran',
    presetSholat: '🤲 Bacaan Sholat',
    presetDoa: '🕊️ Doa Harian',
    s1: 'Dasar game',
    fTitle: 'Judul game',
    phTitle: 'mis. Tantangan Sahabat Rasul',
    fGenre: 'Jenis game',
    fTheme: 'Tema / topik',
    phTheme: 'mis. sahabat Nabi di Madinah',
    fConcept: 'Ide utama',
    optional: '(opsional)',
    phConcept: 'Jelaskan idemu dalam 1–3 kalimat: apa yang dilakukan pemain dan kenapa game ini seru?',
    fAudience: 'Target pemain',
    fPurpose: 'Tujuan',
    fObjective: 'Tujuan pembelajaran',
    phObjective: 'mis. Siswa dapat mengingat perkalian 6–9 dalam 5 detik',
    s2: 'Permainan',
    fLevels: 'Jumlah level',
    fLives: 'Nyawa awal',
    fSession: 'Durasi bermain',
    fDifficulty: 'Tingkat kesulitan awal',
    easy: 'Mudah', normal: 'Normal', hard: 'Sulit',
    s3: 'Fitur',
    s4: 'Tampilan & nuansa',
    fStyle: 'Gaya visual',
    fPalette: 'Palet warna',
    cPrimary: 'Utama', cAccent: 'Aksen', cBg: 'Latar',
    fGameLang: 'Bahasa di dalam game',
    fMood: 'Suasana',
    s5: 'Prompt & hasil kode',
    fPromptLang: 'Bahasa prompt',
    promptLangHint: 'Bahasa prompt yang dikirim ke Gemini. Boleh berbeda dengan bahasa di dalam game.',
    fmtSingle: 'Satu file HTML',
    fmtSingleDesc: 'HTML, CSS, dan JS dalam satu <code>index.html</code>. Paling cocok untuk pratinjau di Gemini Canvas.',
    fmtMulti: 'File terpisah',
    fmtMultiDesc: '<code>index.html</code>, <code>style.css</code>, dan <code>script.js</code>, tersusun seperti folder proyek untuk localhost.',
    optComments: 'Kode dengan komentar yang jelas',
    optLocalhost: 'Sertakan cara menjalankan di localhost',
    optNext: 'Sarankan 3 ide pengembangan',
    fExtra: 'Permintaan tambahan',
    phExtra: "Hal lain, mis. 'Tampilkan nama madrasah kami di layar judul', 'Tambahkan Bismillah sebelum game dimulai', 'Sertakan hiasan kaligrafi Arab'",
    outAria: 'Prompt yang dihasilkan',
    outTitle: 'Prompt STAR kamu',
    legend: ['S · Situasi', 'T · Tugas', 'A · Aksi', 'R · Hasil'],
    copy: '📋 Salin prompt',
    download: '⬇️ Unduh .txt',
    openGemini: '✨ Buka Gemini',
    howtoTitle: 'Cara memakainya di Gemini Canvas',
    howtoSteps: `<li>Klik <strong>Salin prompt</strong>, lalu <strong>Buka Gemini</strong>.</li>
      <li>Di kolom prompt Gemini, aktifkan <strong>Canvas</strong>.</li>
      <li>Tempel prompt lalu kirim. Gemini akan menulis kodenya dan menampilkan pratinjau langsung.</li>
      <li>Mainkan game di pratinjau. Untuk mengubah sesuatu, minta lewat chat (lihat prompt lanjutan di bawah).</li>
      <li>Untuk menjalankan di localhost: simpan kode sebagai <code>index.html</code> di sebuah folder, jalankan <code>python3 -m http.server 8000</code> di folder itu, lalu buka <code>http://localhost:8000</code>.</li>`,
    followTitle: 'Prompt lanjutan untuk terus menyempurnakan',
    followHint: 'Klik untuk menyalin',
    words: 'kata', chars: 'karakter',
    toastCopied: 'Prompt tersalin. Tempel ke Gemini Canvas.',
    toastFollow: 'Prompt lanjutan tersalin',
    toastPreset: (t) => `"${t}" dimuat`,
    fContext: 'Konteks pengetahuan',
    dropText: 'Klik atau seret file ke sini',
    contextLoaded: '✅ Konteks dimuat',
    contextClear: 'Hapus',
    contextUse: 'Sertakan konteks unggahan dalam prompt',
    contextSection: 'Materi referensi (dari dokumen yang diunggah)',
    contextParsing: 'Membaca file...',
  },
};

// ---------- Data ----------

const CANVAS_TECH = {
  en: 'an HTML5 <canvas> rendered with a requestAnimationFrame game loop that uses delta time, so speed is identical on 60 Hz and 120 Hz screens',
  id: 'HTML5 <canvas> yang dirender dengan game loop requestAnimationFrame berbasis delta time, agar kecepatannya sama di layar 60 Hz maupun 120 Hz',
};
const DOM_TECH = {
  en: 'semantic HTML elements styled with CSS (transitions, transforms, grid/flexbox) and driven by vanilla JavaScript',
  id: 'elemen HTML semantik yang diberi gaya dengan CSS (transition, transform, grid/flexbox) dan dijalankan dengan JavaScript murni (vanilla)',
};

const GENRES = {
  quiz: {
    label: { en: 'Quiz / Trivia', id: 'Kuis / Trivia' },
    desc: { en: 'Multiple-choice questions with instant feedback. Great for learning.', id: 'Soal pilihan ganda dengan umpan balik langsung. Cocok untuk belajar.' },
    tech: DOM_TECH,
    mechanics: {
      en: [
        'Show one multiple-choice question at a time with 4 answer options in a shuffled order.',
        'Give instant feedback: highlight the correct answer in green and a wrong pick in red, plus a one-sentence explanation.',
        'Award points for correct answers, with a speed bonus when the timer is on and a streak multiplier for consecutive correct answers.',
        'Group questions into levels of increasing difficulty and never repeat a question within one game.',
      ],
      id: [
        'Tampilkan satu soal pilihan ganda setiap kali, dengan 4 pilihan jawaban yang urutannya diacak.',
        'Beri umpan balik langsung: tandai jawaban benar dengan hijau dan pilihan salah dengan merah, disertai penjelasan satu kalimat.',
        'Beri poin untuk jawaban benar, bonus kecepatan jika timer aktif, dan pengali beruntun untuk jawaban benar berturut-turut.',
        'Kelompokkan soal ke dalam level yang makin sulit, dan jangan ulangi soal yang sama dalam satu permainan.',
      ],
    },
    controls: {
      en: 'Mouse/touch to tap an answer; keyboard keys 1–4 to choose an answer and Enter to continue.',
      id: 'Mouse/sentuh untuk memilih jawaban; tombol keyboard 1–4 untuk memilih dan Enter untuk lanjut.',
    },
    content: {
      en: (c) => `Write a question bank of at least ${c.levels * 6} original, factually accurate questions about ${c.theme}, stored as a JavaScript array of objects { question, options: [4 strings], answerIndex, explanation, level }. Spread them evenly across ${n('en', c.levels, 'level')}, from easiest to hardest.`,
      id: (c) => `Tulis bank soal berisi minimal ${c.levels * 6} soal orisinal yang faktanya akurat tentang ${c.theme}, disimpan sebagai array objek JavaScript { question, options: [4 string], answerIndex, explanation, level }. Bagi rata ke ${c.levels} level, dari yang termudah sampai tersulit.`,
    },
  },
  memory: {
    label: { en: 'Memory Card Match', id: 'Mencocokkan Kartu (Memori)' },
    desc: { en: 'Flip cards to find matching pairs. Simple, calm, works for any topic.', id: 'Buka kartu untuk menemukan pasangannya. Sederhana, santai, cocok untuk topik apa pun.' },
    tech: DOM_TECH,
    mechanics: {
      en: [
        'Show a grid of face-down cards; the player flips two at a time. Matching pairs stay face-up; non-matching pairs flip back after 0.8 s.',
        'Block input while two cards are being compared so cards cannot be flipped too fast.',
        'Track moves and time; award 1–3 stars at the end based on how few moves were used.',
        'Each level increases the grid size (for example 4×3, then 4×4, then 6×4).',
        'Use a smooth 3D card-flip animation with CSS transforms (backface-visibility).',
      ],
      id: [
        'Tampilkan kisi kartu tertutup; pemain membuka dua kartu sekaligus. Pasangan yang cocok tetap terbuka; yang tidak cocok tertutup lagi setelah 0,8 detik.',
        'Kunci input selama dua kartu sedang dibandingkan agar kartu tidak bisa dibuka terlalu cepat.',
        'Catat jumlah langkah dan waktu; beri 1–3 bintang di akhir berdasarkan sedikitnya langkah.',
        'Setiap level memperbesar ukuran kisi (misalnya 4×3, lalu 4×4, lalu 6×4).',
        'Gunakan animasi balik kartu 3D yang halus dengan CSS transform (backface-visibility).',
      ],
    },
    controls: {
      en: 'Mouse/touch to flip cards; arrow keys to move focus and Enter/Space to flip.',
      id: 'Mouse/sentuh untuk membuka kartu; tombol panah untuk memindah fokus dan Enter/Spasi untuk membuka.',
    },
    content: {
      en: (c) => `Create at least ${Math.max(12, c.levels * 6)} matching pairs related to ${c.theme} (for example term ↔ meaning, picture/emoji ↔ word, or question ↔ answer), stored as a JavaScript array.`,
      id: (c) => `Buat minimal ${Math.max(12, c.levels * 6)} pasangan yang berkaitan dengan ${c.theme} (misalnya istilah ↔ arti, gambar/emoji ↔ kata, atau soal ↔ jawaban), disimpan sebagai array JavaScript.`,
    },
  },
  platformer: {
    label: { en: 'Platformer', id: 'Platformer (lompat-lompatan)' },
    desc: { en: 'Run, jump and collect items across side-scrolling levels.', id: 'Lari, lompat, dan kumpulkan item di level yang bergulir ke samping.' },
    tech: CANVAS_TECH,
    mechanics: {
      en: [
        'The player runs and jumps with gravity, acceleration and friction; add coyote time (100 ms) and jump buffering so the controls feel tight.',
        'Each level has platforms, collectibles, hazards and a goal flag; define levels as data (tile-map arrays) so new levels are easy to add.',
        'The camera smoothly follows the player horizontally.',
        'Simple enemies patrol back and forth; jumping on them defeats them, touching them from the side costs a life.',
        'Use axis-separated AABB collision so the player never gets stuck in walls.',
      ],
      id: [
        'Pemain berlari dan melompat dengan gravitasi, akselerasi, dan gesekan; tambahkan coyote time (100 ms) dan jump buffering agar kontrol terasa responsif.',
        'Setiap level punya platform, item untuk dikumpulkan, rintangan, dan bendera finis; definisikan level sebagai data (array tile-map) agar level baru mudah ditambahkan.',
        'Kamera mengikuti pemain secara horizontal dengan halus.',
        'Musuh sederhana berpatroli bolak-balik; melompat di atasnya mengalahkannya, menyentuhnya dari samping mengurangi satu nyawa.',
        'Gunakan deteksi tabrakan AABB per sumbu agar pemain tidak pernah tersangkut di dinding.',
      ],
    },
    controls: {
      en: 'Arrow keys or A/D to move, Space/W/Up to jump, P or Esc to pause. On touch screens, show on-screen left/right/jump buttons.',
      id: 'Tombol panah atau A/D untuk bergerak, Spasi/W/Atas untuk melompat, P atau Esc untuk jeda. Di layar sentuh, tampilkan tombol kiri/kanan/lompat di layar.',
    },
    content: {
      en: (c) => `Design ${n('en', c.levels, 'handcrafted level')} themed around ${c.theme}, each slightly longer and harder than the previous one, and introduce one new idea per level.`,
      id: (c) => `Rancang ${c.levels} level buatan tangan bertema ${c.theme}, masing-masing sedikit lebih panjang dan sulit dari sebelumnya, dengan satu ide baru di setiap level.`,
    },
  },
  arcade: {
    label: { en: 'Arcade Catch & Dodge', id: 'Arkade Tangkap & Hindari' },
    desc: { en: 'Catch the good items, dodge the bad ones. Quick and addictive.', id: 'Tangkap item yang baik, hindari yang buruk. Cepat dan bikin ketagihan.' },
    tech: CANVAS_TECH,
    mechanics: {
      en: [
        'The player moves horizontally along the bottom of the screen; items fall from the top.',
        'Catching good items earns points; touching bad items costs a life.',
        'Spawn rate and falling speed increase over time and with each level.',
        'Occasionally spawn power-ups (shield, magnet, slow motion) that last a few seconds and are shown in the HUD.',
        'Consecutive catches build a combo multiplier that resets when an item is missed.',
      ],
      id: [
        'Pemain bergerak horizontal di bagian bawah layar; item berjatuhan dari atas.',
        'Menangkap item baik menambah poin; menyentuh item buruk mengurangi satu nyawa.',
        'Frekuensi munculnya item dan kecepatan jatuhnya meningkat seiring waktu dan di setiap level.',
        'Sesekali munculkan power-up (perisai, magnet, gerak lambat) yang bertahan beberapa detik dan ditampilkan di HUD.',
        'Tangkapan berturut-turut menaikkan pengali kombo yang kembali ke awal jika ada item terlewat.',
      ],
    },
    controls: {
      en: 'Arrow keys or A/D to move, or move the mouse; on touch screens, drag anywhere to move. P or Esc to pause.',
      id: 'Tombol panah atau A/D untuk bergerak, atau gerakkan mouse; di layar sentuh, geser di mana saja. P atau Esc untuk jeda.',
    },
    content: {
      en: (c) => `Theme every falling item, the player character and the background around ${c.theme}, with at least 4 kinds of good items and 3 kinds of bad items.`,
      id: (c) => `Sesuaikan semua item yang jatuh, karakter pemain, dan latar dengan tema ${c.theme}, dengan minimal 4 jenis item baik dan 3 jenis item buruk.`,
    },
  },
  adventure: {
    label: { en: 'Choose-Your-Own Adventure', id: 'Petualangan Pilih Jalan Cerita' },
    desc: { en: "A branching story where the player's choices change the ending.", id: 'Cerita bercabang; pilihan pemain menentukan akhir cerita.' },
    tech: DOM_TECH,
    mechanics: {
      en: [
        'Present the story as scenes: a short paragraph of text, a simple illustration (CSS, SVG or emoji) and 2–4 choice buttons.',
        'Choices change player stats (for example health, courage, reputation) and items in an inventory, shown in a side panel.',
        'Some choices are only available when the player has a certain item or a high enough stat.',
        'Include at least 3 different endings (good, neutral, bad) and show which endings have been found.',
        'Animate text appearing with a typewriter effect that can be skipped by clicking.',
      ],
      id: [
        'Sajikan cerita dalam adegan: satu paragraf pendek, ilustrasi sederhana (CSS, SVG, atau emoji), dan 2–4 tombol pilihan.',
        'Pilihan mengubah statistik pemain (misalnya kesehatan, keberanian, reputasi) dan isi inventaris, yang ditampilkan di panel samping.',
        'Beberapa pilihan hanya tersedia jika pemain memiliki item tertentu atau statistik yang cukup tinggi.',
        'Sediakan minimal 3 akhir cerita berbeda (baik, netral, buruk) dan tampilkan akhir mana yang sudah ditemukan.',
        'Munculkan teks dengan efek mesin ketik yang bisa dilewati dengan sekali klik.',
      ],
    },
    controls: {
      en: 'Mouse/touch to pick a choice; keyboard keys 1–4 to choose and Space to skip the text animation.',
      id: 'Mouse/sentuh untuk memilih; tombol keyboard 1–4 untuk memilih dan Spasi untuk melewati animasi teks.',
    },
    content: {
      en: (c) => `Write the complete branching story about ${c.theme} as a JavaScript object of scene nodes { id, text, choices: [{ label, next, effects, requires }] }, with at least ${Math.max(12, c.levels * 5)} scenes split into ${n('en', c.levels, 'chapter')}.`,
      id: (c) => `Tulis cerita bercabang yang lengkap tentang ${c.theme} sebagai objek JavaScript berisi node adegan { id, text, choices: [{ label, next, effects, requires }] }, dengan minimal ${Math.max(12, c.levels * 5)} adegan yang dibagi ke dalam ${c.levels} bab.`,
    },
  },
  match3: {
    label: { en: 'Match-3 Puzzle', id: 'Puzzle Match-3' },
    desc: { en: 'Swap tiles to line up three or more. Classic puzzle fun.', id: 'Tukar ubin untuk menyusun tiga atau lebih. Puzzle klasik yang seru.' },
    tech: CANVAS_TECH,
    mechanics: {
      en: [
        'An 8×8 board of tiles in 5–6 types; swap two neighbouring tiles to make a line of 3 or more.',
        'Matched tiles disappear, tiles above fall down, and new tiles drop in; chain reactions score extra points.',
        'Swaps that make no match animate back to their original place.',
        'Matching 4 creates a line-clear tile, and matching 5 creates a colour bomb.',
        'Each level has a goal (reach a score or clear a number of a tile type) within a limited number of moves.',
        'Make sure the starting board has no matches and at least one possible move; reshuffle if no moves remain.',
      ],
      id: [
        'Papan 8×8 berisi 5–6 jenis ubin; tukar dua ubin yang bersebelahan untuk membentuk barisan 3 atau lebih.',
        'Ubin yang cocok menghilang, ubin di atasnya turun, dan ubin baru berjatuhan; reaksi berantai memberi poin tambahan.',
        'Tukaran yang tidak menghasilkan kecocokan kembali ke posisi semula dengan animasi.',
        'Mencocokkan 4 membuat ubin penghapus baris, dan mencocokkan 5 membuat bom warna.',
        'Setiap level punya target (capai skor tertentu atau hapus sejumlah ubin jenis tertentu) dengan jumlah langkah terbatas.',
        'Pastikan papan awal tidak punya kecocokan dan ada minimal satu langkah yang mungkin; acak ulang jika tidak ada langkah tersisa.',
      ],
    },
    controls: {
      en: 'Click/tap a tile and then a neighbour, or swipe on touch screens. Arrow keys to move a cursor and Enter to select.',
      id: 'Klik/ketuk satu ubin lalu ubin sebelahnya, atau geser di layar sentuh. Tombol panah untuk memindah kursor dan Enter untuk memilih.',
    },
    content: {
      en: (c) => `Design the tile icons around ${c.theme} and define ${n('en', c.levels, 'level goal')} as data, with difficulty that rises steadily.`,
      id: (c) => `Rancang ikon ubin bertema ${c.theme} dan definisikan target untuk ${c.levels} level sebagai data, dengan tingkat kesulitan yang naik secara bertahap.`,
    },
  },
  word: {
    label: { en: 'Word Guess', id: 'Tebak Kata' },
    desc: { en: 'Guess the hidden word letter by letter, with hints. Good for vocabulary.', id: 'Tebak kata tersembunyi huruf demi huruf, dengan petunjuk. Cocok untuk kosakata.' },
    tech: DOM_TECH,
    mechanics: {
      en: [
        'Show a hidden word as blank tiles; the player guesses one letter at a time using an on-screen keyboard or a physical keyboard.',
        'Correct letters are revealed in every position; wrong guesses use up one of a limited number of attempts, shown with a visual meter.',
        'Show the category and offer a hint (a definition or clue) after 3 wrong guesses.',
        'Letters already guessed are disabled on the on-screen keyboard.',
        'Score depends on word length and how few wrong guesses were made.',
      ],
      id: [
        'Tampilkan kata tersembunyi sebagai kotak kosong; pemain menebak satu huruf setiap kali dengan keyboard di layar atau keyboard fisik.',
        'Huruf yang benar terbuka di semua posisinya; tebakan salah mengurangi jatah percobaan yang terbatas, ditampilkan dengan indikator visual.',
        'Tampilkan kategori dan berikan petunjuk (definisi atau clue) setelah 3 tebakan salah.',
        'Huruf yang sudah ditebak dinonaktifkan di keyboard layar.',
        'Skor bergantung pada panjang kata dan sedikitnya tebakan salah.',
      ],
    },
    controls: {
      en: 'Physical keyboard letters or tapping the on-screen keyboard; Enter to go to the next word.',
      id: 'Huruf di keyboard fisik atau ketuk keyboard di layar; Enter untuk lanjut ke kata berikutnya.',
    },
    content: {
      en: (c) => `Create a word list of at least ${c.levels * 8} words related to ${c.theme}, each with a clue, grouped into ${n('en', c.levels, 'level')} by word length and difficulty.`,
      id: (c) => `Buat daftar minimal ${c.levels * 8} kata yang berkaitan dengan ${c.theme}, masing-masing dengan petunjuk, dikelompokkan ke dalam ${c.levels} level berdasarkan panjang dan tingkat kesulitan kata.`,
    },
  },
  snake: {
    label: { en: 'Snake', id: 'Ular (Snake)' },
    desc: { en: "The classic grid game: eat, grow longer and don't hit yourself.", id: 'Game klasik: makan, tumbuh makin panjang, dan jangan menabrak diri sendiri.' },
    tech: CANVAS_TECH,
    mechanics: {
      en: [
        'Grid-based snake moving at a fixed tick rate; eating food makes it grow by one segment.',
        "Hitting a wall or the snake's own body ends the round (or costs a life).",
        'Speed increases slightly every few pieces of food and with each level.',
        'Special bonus food appears for a few seconds and is worth extra points.',
        'Queue direction changes so that fast key presses never make the snake reverse into itself.',
      ],
      id: [
        'Ular bergerak di kisi dengan kecepatan tick tetap; makan makanan membuatnya bertambah satu ruas.',
        'Menabrak dinding atau tubuhnya sendiri mengakhiri ronde (atau mengurangi satu nyawa).',
        'Kecepatan bertambah sedikit setiap beberapa makanan dan di setiap level.',
        'Makanan bonus khusus muncul beberapa detik dan bernilai poin ekstra.',
        'Antrekan perubahan arah agar tombol yang ditekan cepat tidak membuat ular berbalik menabrak dirinya.',
      ],
    },
    controls: {
      en: 'Arrow keys or WASD to steer; swipe on touch screens; P or Esc to pause.',
      id: 'Tombol panah atau WASD untuk mengarahkan; geser di layar sentuh; P atau Esc untuk jeda.',
    },
    content: {
      en: (c) => `Theme the snake, food and board around ${c.theme}, and from level 2 onwards add some obstacle walls (${n('en', c.levels, 'level')} in total).`,
      id: (c) => `Sesuaikan ular, makanan, dan papan dengan tema ${c.theme}, dan mulai level 2 tambahkan beberapa dinding penghalang (total ${c.levels} level).`,
    },
  },
  clicker: {
    label: { en: 'Idle / Clicker', id: 'Idle / Clicker' },
    desc: { en: 'Tap to earn, buy upgrades, watch the numbers grow.', id: 'Ketuk untuk mendapat koin, beli upgrade, lihat angkanya terus naik.' },
    tech: DOM_TECH,
    mechanics: {
      en: [
        'Clicking a large main object earns currency, with a satisfying bounce and a floating "+N" number.',
        'A shop sells upgrades that increase per-click value and generators that earn currency automatically every second.',
        'Upgrade prices grow exponentially (around ×1.15 per purchase); show which upgrades are affordable.',
        'Format big numbers neatly (1.2K, 3.4M, 5.6B).',
        'Milestones unlock new upgrade tiers; levels act as milestone stages.',
        'Auto-save progress to localStorage every 10 seconds and give offline earnings when the player returns.',
      ],
      id: [
        'Mengklik objek utama yang besar menghasilkan mata uang, dengan efek memantul dan angka "+N" yang melayang.',
        'Toko menjual upgrade yang menaikkan nilai per klik dan generator yang menghasilkan mata uang otomatis setiap detik.',
        'Harga upgrade naik secara eksponensial (sekitar ×1,15 per pembelian); tandai upgrade yang sudah terjangkau.',
        'Format angka besar dengan rapi (1,2 rb, 3,4 jt, 5,6 M).',
        'Pencapaian tertentu membuka tingkat upgrade baru; level berfungsi sebagai tahap pencapaian.',
        'Simpan progres otomatis ke localStorage setiap 10 detik dan beri penghasilan offline saat pemain kembali.',
      ],
    },
    controls: {
      en: 'Mouse/touch to click and buy; Space to click the main object.',
      id: 'Mouse/sentuh untuk mengklik dan membeli; Spasi untuk mengklik objek utama.',
    },
    content: {
      en: (c) => `Name and theme the currency, the main object and at least ${Math.max(8, c.levels * 3)} upgrades and generators around ${c.theme}.`,
      id: (c) => `Beri nama dan tema untuk mata uang, objek utama, serta minimal ${Math.max(8, c.levels * 3)} upgrade dan generator sesuai tema ${c.theme}.`,
    },
  },
  custom: {
    label: { en: 'Custom (describe your own)', id: 'Kustom (jelaskan sendiri)' },
    desc: { en: 'Describe your game in "Core idea" and the prompt will be built around it.', id: 'Jelaskan game-mu di kolom "Ide utama", dan prompt akan disusun berdasarkan itu.' },
    tech: {
      en: 'whichever fits the game best: an HTML5 <canvas> with a requestAnimationFrame loop for real-time action, or DOM elements with CSS for turn-based or card games',
      id: 'teknologi yang paling cocok: HTML5 <canvas> dengan loop requestAnimationFrame untuk aksi real-time, atau elemen DOM dengan CSS untuk game bergiliran atau kartu',
    },
    mechanics: {
      en: [
        'Turn the core idea into a clear core loop: what the player does every few seconds, what they are rewarded with, and what makes it harder over time.',
        'Define clear win and lose conditions.',
      ],
      id: [
        'Ubah ide utama menjadi core loop yang jelas: apa yang dilakukan pemain setiap beberapa detik, hadiah apa yang didapat, dan apa yang membuatnya makin sulit seiring waktu.',
        'Tentukan kondisi menang dan kalah yang jelas.',
      ],
    },
    controls: {
      en: 'Choose controls that suit the game; support mouse/touch and keyboard.',
      id: 'Pilih kontrol yang sesuai dengan game; dukung mouse/sentuh dan keyboard.',
    },
    content: {
      en: (c) => `Create all the content the game needs (levels, items, questions or story) themed around ${c.theme}, enough for ${n('en', c.levels, 'level')}.`,
      id: (c) => `Buat semua konten yang dibutuhkan game (level, item, soal, atau cerita) bertema ${c.theme}, cukup untuk ${c.levels} level.`,
    },
  },
};

const AUDIENCES = {
  kids: {
    label: { en: 'Kids (6–9 years)', id: 'Anak-anak (6–9 tahun)' },
    who: { en: 'children aged 6–9', id: 'anak-anak usia 6–9 tahun' },
    ux: {
      en: 'Use very large buttons, few words per screen, simple vocabulary, generous time limits, lots of positive encouragement, and nothing scary or violent.',
      id: 'Gunakan tombol yang sangat besar, sedikit kata per layar, kosakata sederhana, batas waktu yang longgar, banyak pujian positif, dan tanpa unsur menakutkan atau kekerasan.',
    },
  },
  preteens: {
    label: { en: 'Pre-teens (10–13 years)', id: 'Praremaja (10–13 tahun)' },
    who: { en: 'pre-teens aged 10–13', id: 'praremaja usia 10–13 tahun' },
    ux: {
      en: 'Keep instructions short and visual, use encouraging feedback, and make the difficulty curve gentle but rewarding.',
      id: 'Buat instruksi singkat dan visual, beri umpan balik yang menyemangati, dan buat kenaikan tingkat kesulitan yang landai tapi memuaskan.',
    },
  },
  teens: {
    label: { en: 'Teens (14–17 years)', id: 'Remaja (14–17 tahun)' },
    who: { en: 'teenagers aged 14–17', id: 'remaja usia 14–17 tahun' },
    ux: {
      en: 'Use a modern, stylish look, a real challenge, and rewards that are worth bragging about.',
      id: 'Gunakan tampilan modern yang keren, tantangan yang sungguhan, dan hadiah yang layak dipamerkan.',
    },
  },
  adults: {
    label: { en: 'Adults', id: 'Dewasa' },
    who: { en: 'adults', id: 'orang dewasa' },
    ux: {
      en: "Keep the interface clean and efficient, respect the player's time, and let them skip tutorials.",
      id: 'Buat antarmuka yang bersih dan efisien, hargai waktu pemain, dan izinkan mereka melewati tutorial.',
    },
  },
  all: {
    label: { en: 'All ages', id: 'Semua umur' },
    who: { en: 'players of all ages', id: 'pemain segala usia' },
    ux: {
      en: 'Make it easy to pick up, with clear instructions and a difficulty that can be adjusted.',
      id: 'Buat mudah dimainkan siapa saja, dengan instruksi yang jelas dan tingkat kesulitan yang bisa diatur.',
    },
  },
};

const PURPOSES = {
  fun: {
    label: { en: 'Entertainment', id: 'Hiburan' },
    text: { en: 'pure entertainment, so it should be fun, replayable and satisfying', id: 'hiburan, jadi game harus seru, bikin ingin main lagi, dan memuaskan' },
  },
  education: {
    label: { en: 'Education / learning', id: 'Pendidikan / pembelajaran' },
    text: { en: 'education, so players should learn while playing, and the learning content must be accurate', id: 'pendidikan, jadi pemain harus belajar sambil bermain, dan materi pembelajarannya harus akurat' },
  },
  training: {
    label: { en: 'Corporate training', id: 'Pelatihan perusahaan' },
    text: { en: 'corporate training, so it should reinforce knowledge in an engaging, professional way', id: 'pelatihan perusahaan, jadi game harus memperkuat pengetahuan dengan cara yang menarik dan profesional' },
  },
  marketing: {
    label: { en: 'Marketing / brand engagement', id: 'Pemasaran / promosi brand' },
    text: { en: 'marketing and brand engagement, so it should be shareable, quick to play and memorable', id: 'pemasaran dan promosi brand, jadi game harus mudah dibagikan, cepat dimainkan, dan mudah diingat' },
  },
};

const SESSIONS = {
  3: { label: { en: '~3 minutes', id: '~3 menit' }, text: { en: 'about 3 minutes', id: 'sekitar 3 menit' } },
  5: { label: { en: '~5 minutes', id: '~5 menit' }, text: { en: 'about 5 minutes', id: 'sekitar 5 menit' } },
  10: { label: { en: '~10 minutes', id: '~10 menit' }, text: { en: 'about 10 minutes', id: 'sekitar 10 menit' } },
  20: { label: { en: '20+ minutes', id: '20+ menit' }, text: { en: '20 minutes or more', id: '20 menit atau lebih' } },
};

const STYLES = {
  flat: {
    label: { en: 'Flat & modern', id: 'Flat & modern' },
    text: { en: 'flat, modern design with rounded corners, soft shadows and clean geometric shapes', id: 'desain flat dan modern dengan sudut membulat, bayangan lembut, dan bentuk geometris yang rapi' },
  },
  pixel: {
    label: { en: 'Retro pixel art', id: 'Pixel art retro' },
    text: { en: 'retro pixel-art look (crisp pixels with image-rendering: pixelated, chunky sprites drawn in code, and a pixel-style font from Google Fonts such as "Press Start 2P")', id: 'tampilan pixel art retro (piksel tajam dengan image-rendering: pixelated, sprite besar yang digambar lewat kode, dan font gaya piksel dari Google Fonts seperti "Press Start 2P")' },
  },
  cartoon: {
    label: { en: 'Bright cartoon', id: 'Kartun ceria' },
    text: { en: 'bright, playful cartoon style with thick outlines, bouncy animations and expressive characters', id: 'gaya kartun yang cerah dan jenaka dengan garis tepi tebal, animasi memantul, dan karakter yang ekspresif' },
  },
  neon: {
    label: { en: 'Neon / cyberpunk', id: 'Neon / cyberpunk' },
    text: { en: 'neon cyberpunk style with a dark background, glowing outlines (box-shadow / shadowBlur), and scanline or grid effects', id: 'gaya neon cyberpunk dengan latar gelap, garis tepi menyala (box-shadow / shadowBlur), dan efek scanline atau kisi' },
  },
  minimal: {
    label: { en: 'Minimalist', id: 'Minimalis' },
    text: { en: 'minimalist style with lots of white space, a limited colour set and elegant typography', id: 'gaya minimalis dengan banyak ruang kosong, sedikit warna, dan tipografi yang elegan' },
  },
  paper: {
    label: { en: 'Hand-drawn / paper', id: 'Gambar tangan / kertas' },
    text: { en: 'hand-drawn paper style with a paper-texture background (CSS gradients), sketchy borders and a handwriting font from Google Fonts', id: 'gaya gambar tangan di atas kertas dengan latar bertekstur kertas (gradien CSS), garis tepi seperti sketsa, dan font tulisan tangan dari Google Fonts' },
  },
  fantasy: {
    label: { en: 'Fantasy / storybook', id: 'Fantasi / buku dongeng' },
    text: { en: 'fantasy storybook style with warm colours, ornate frames, parchment panels and a serif display font', id: 'gaya buku dongeng fantasi dengan warna hangat, bingkai berornamen, panel seperti perkamen, dan font serif dekoratif' },
  },
};

const PALETTES = {
  auto: { label: { en: 'Let Gemini choose', id: 'Biarkan Gemini memilih' }, colors: [] },
  ocean: { label: { en: 'Ocean', id: 'Samudra' }, colors: ['#0ea5e9', '#facc15', '#0b1e3f', '#e0f2fe'] },
  sunset: { label: { en: 'Sunset', id: 'Senja' }, colors: ['#f43f5e', '#fb923c', '#2a1033', '#fff1e6'] },
  forest: { label: { en: 'Forest', id: 'Hutan' }, colors: ['#16a34a', '#eab308', '#0f2417', '#ecfccb'] },
  candy: { label: { en: 'Candy', id: 'Permen' }, colors: ['#ec4899', '#8b5cf6', '#fff5fb', '#38bdf8'] },
  neon: { label: { en: 'Neon', id: 'Neon' }, colors: ['#22d3ee', '#f0abfc', '#0a0a1a', '#a3e635'] },
  earth: { label: { en: 'Earth', id: 'Tanah' }, colors: ['#b45309', '#65a30d', '#fdf6e3', '#3f2a14'] },
  mono: { label: { en: 'Monochrome', id: 'Monokrom' }, colors: ['#111827', '#6b7280', '#ffffff', '#ef4444'] },
  custom: { label: { en: 'Custom colours…', id: 'Warna sendiri…' }, colors: [] },
};

const GAME_LANGS = {
  en: { label: { en: 'English', id: 'Bahasa Inggris' }, text: { en: 'English', id: 'bahasa Inggris' } },
  id: { label: { en: 'Bahasa Indonesia', id: 'Bahasa Indonesia' }, text: { en: 'Bahasa Indonesia', id: 'bahasa Indonesia' } },
  bi: {
    label: { en: 'Bilingual EN / ID (toggle)', id: 'Dua bahasa EN / ID (tombol ganti)' },
    text: { en: 'both English and Bahasa Indonesia, with a language toggle button on every screen (store all text in one translations object)', id: 'bahasa Inggris dan bahasa Indonesia, dengan tombol ganti bahasa di setiap layar (simpan semua teks dalam satu objek terjemahan)' },
  },
  ar: { label: { en: 'Arabic (RTL)', id: 'Bahasa Arab (RTL)' }, text: { en: 'Arabic, with a right-to-left layout', id: 'bahasa Arab, dengan tata letak kanan-ke-kiri' } },
  es: { label: { en: 'Spanish', id: 'Bahasa Spanyol' }, text: { en: 'Spanish', id: 'bahasa Spanyol' } },
  ja: { label: { en: 'Japanese', id: 'Bahasa Jepang' }, text: { en: 'Japanese', id: 'bahasa Jepang' } },
};

const MOODS = {
  cheerful: { label: { en: 'Cheerful & encouraging', id: 'Ceria & menyemangati' }, text: { en: 'cheerful and encouraging', id: 'ceria dan menyemangati' } },
  calm: { label: { en: 'Calm & relaxing', id: 'Tenang & santai' }, text: { en: 'calm and relaxing', id: 'tenang dan santai' } },
  exciting: { label: { en: 'Fast & exciting', id: 'Cepat & seru' }, text: { en: 'fast and exciting', id: 'cepat dan seru' } },
  mysterious: { label: { en: 'Mysterious', id: 'Misterius' }, text: { en: 'mysterious and suspenseful', id: 'misterius dan menegangkan' } },
  professional: { label: { en: 'Professional', id: 'Profesional' }, text: { en: 'professional and clean', id: 'profesional dan rapi' } },
};

const SELECT_DATA = { AUDIENCES, PURPOSES, SESSIONS, STYLES, PALETTES, GAME_LANGS, MOODS };
const SELECT_DEFAULTS = { audience: 'all', purpose: 'fun', session: '5', style: 'flat', palette: 'auto', lang: 'en', mood: 'cheerful' };

const DIFFICULTY = { easy: { en: 'Easy', id: 'Mudah' }, normal: { en: 'Normal', id: 'Normal' }, hard: { en: 'Hard', id: 'Sulit' } };

// Each feature: [id, label, default on?, instruction builder (c, L) => string]
const FEATURES = [
  ['score', { en: 'Score & points popups', id: 'Skor & angka poin melayang' }, true, () => T({
    en: 'Score system shown in the HUD, with animated "+points" popups.',
    id: 'Sistem skor yang tampil di HUD, dengan animasi angka "+poin" yang muncul.',
  })],
  ['highscore', { en: 'Saved high score', id: 'Skor tertinggi tersimpan' }, true, () => T({
    en: 'Save the best score (per difficulty) in localStorage and show it on the start and game-over screens. Wrap every localStorage call in try/catch so the game still works when storage is blocked (for example in a sandboxed preview).',
    id: 'Simpan skor tertinggi (per tingkat kesulitan) di localStorage dan tampilkan di layar awal dan layar game over. Bungkus setiap pemanggilan localStorage dengan try/catch agar game tetap berjalan saat penyimpanan diblokir (misalnya di pratinjau sandbox).',
  })],
  ['timer', { en: 'Timer / countdown', id: 'Timer / hitung mundur' }, false, () => T({
    en: 'A visible timer in the HUD; running out of time ends the round or level, and the bar changes colour when time is almost up.',
    id: 'Timer yang terlihat di HUD; jika waktu habis, ronde atau level berakhir, dan warna bar berubah saat waktu hampir habis.',
  })],
  ['lives', { en: 'Lives / hearts', id: 'Nyawa / hati' }, true, (c) => T({
    en: `Lives system: the player starts with ${n('en', c.lives, 'life', 'lives')}, shown as hearts; losing them all ends the game.`,
    id: `Sistem nyawa: pemain mulai dengan ${c.lives} nyawa yang ditampilkan sebagai hati; jika semua habis, game berakhir.`,
  })],
  ['levels', { en: 'Level progression', id: 'Tingkatan level' }, true, (c) => c.levels > 1 ? T({
    en: `Level progression across ${c.levels} levels, each harder than the last, with a short "Level X complete" screen between levels that summarises the score.`,
    id: `Tingkatan ${c.levels} level, masing-masing lebih sulit dari sebelumnya, dengan layar singkat "Level X selesai" di antara level yang merangkum skor.`,
  }) : T({
    en: 'One complete level with a clear finish and a final results screen.',
    id: 'Satu level lengkap dengan akhir yang jelas dan layar hasil akhir.',
  })],
  ['difficultySelect', { en: 'Difficulty selector', id: 'Pilihan tingkat kesulitan' }, true, (c) => T({
    en: `Difficulty selector (Easy / Normal / Hard, default ${DIFFICULTY[c.difficulty].en}) on the start screen that changes speed, time limits and scoring.`,
    id: `Pilihan tingkat kesulitan (Mudah / Normal / Sulit, bawaan ${DIFFICULTY[c.difficulty].id}) di layar awal yang mengubah kecepatan, batas waktu, dan penilaian.`,
  })],
  ['pause', { en: 'Pause / resume', id: 'Jeda / lanjut' }, true, () => T({
    en: 'Pause and resume with P or Esc and an on-screen pause button; all game logic and timers stop while paused, and the game pauses itself when the browser tab is hidden.',
    id: 'Jeda dan lanjutkan dengan P atau Esc serta tombol jeda di layar; semua logika game dan timer berhenti saat dijeda, dan game otomatis berhenti saat tab browser disembunyikan.',
  })],
  ['tutorial', { en: '"How to play" screen', id: 'Layar "Cara bermain"' }, true, () => T({
    en: '"How to Play" screen with short, illustrated instructions, reachable from the start screen.',
    id: 'Layar "Cara Bermain" dengan instruksi singkat bergambar, yang bisa dibuka dari layar awal.',
  })],
  ['sfx', { en: 'Sound effects', id: 'Efek suara' }, true, () => T({
    en: 'Sound effects generated with the Web Audio API (no audio files): click, success, fail, level up and game over. Include a mute button, and create the AudioContext only after the first user interaction.',
    id: 'Efek suara yang dibuat dengan Web Audio API (tanpa file audio): klik, berhasil, gagal, naik level, dan game over. Sediakan tombol bisu, dan buat AudioContext hanya setelah interaksi pertama pengguna.',
  })],
  ['music', { en: 'Background music', id: 'Musik latar' }, false, () => T({
    en: 'A simple, loopable background melody generated with the Web Audio API, with its own on/off toggle and a low default volume.',
    id: 'Melodi latar sederhana yang berulang, dibuat dengan Web Audio API, dengan tombol nyala/mati sendiri dan volume awal yang pelan.',
  })],
  ['juice', { en: 'Animations & particles', id: 'Animasi & partikel' }, true, () => T({
    en: 'Lively feedback: smooth screen transitions, particle bursts on success, a small screen shake on mistakes and button hover/press animations.',
    id: 'Umpan balik yang hidup: transisi layar yang halus, ledakan partikel saat berhasil, guncangan layar kecil saat salah, dan animasi tombol saat disorot/ditekan.',
  })],
  ['avatar', { en: 'Character selection', id: 'Pilih karakter' }, false, () => T({
    en: "A character/avatar selection screen with at least 4 options (drawn with CSS, SVG, canvas or emoji) that change how the player looks.",
    id: 'Layar pemilihan karakter/avatar dengan minimal 4 pilihan (digambar dengan CSS, SVG, canvas, atau emoji) yang mengubah tampilan pemain.',
  })],
  ['achievements', { en: 'Achievements / badges', id: 'Pencapaian / lencana' }, false, () => T({
    en: 'At least 5 achievements or badges (for example first win, perfect level, speed run) with a pop-up message when one is unlocked and a screen that lists them, saved in localStorage.',
    id: 'Minimal 5 pencapaian atau lencana (misalnya menang pertama, level sempurna, selesai tercepat) dengan pesan pop-up saat terbuka dan layar yang menampilkan semuanya, disimpan di localStorage.',
  })],
  ['hints', { en: 'Hints', id: 'Petunjuk' }, false, () => T({
    en: 'A hint button with a limited number of hints per game, shown in the HUD.',
    id: 'Tombol petunjuk dengan jumlah petunjuk terbatas per permainan, ditampilkan di HUD.',
  })],
  ['leaderboard', { en: 'Local leaderboard', id: 'Papan peringkat lokal' }, false, () => T({
    en: 'A local top-10 leaderboard where the player enters a name, saved in localStorage.',
    id: 'Papan peringkat 10 besar lokal tempat pemain memasukkan nama, disimpan di localStorage.',
  })],
  ['settings', { en: 'Settings menu', id: 'Menu pengaturan' }, false, () => T({
    en: 'A settings menu with sound volume, music on/off, difficulty and a "reset progress" button that asks for confirmation.',
    id: 'Menu pengaturan berisi volume suara, musik nyala/mati, tingkat kesulitan, dan tombol "atur ulang progres" yang meminta konfirmasi.',
  })],
  ['mobile', { en: 'Mobile & touch friendly', id: 'Ramah HP & layar sentuh' }, true, () => T({
    en: 'A responsive layout from 360 px phones to large desktops, with touch controls (taps, swipes or on-screen buttons) whose touch targets are at least 44 px; avoid unwanted page scrolling or zooming while playing.',
    id: 'Tata letak responsif dari HP 360 px hingga desktop besar, dengan kontrol sentuh (ketuk, geser, atau tombol di layar) yang area sentuhnya minimal 44 px; cegah halaman tergulir atau ter-zoom tanpa sengaja saat bermain.',
  })],
  ['a11y', { en: 'Accessibility', id: 'Aksesibilitas' }, true, () => T({
    en: 'Accessibility: full keyboard control, visible focus outlines, ARIA labels on icon buttons, sufficient colour contrast, colour-blind-safe feedback (never rely on colour alone, so add icons or text), and honour prefers-reduced-motion.',
    id: 'Aksesibilitas: bisa dimainkan penuh dengan keyboard, garis fokus yang terlihat, label ARIA pada tombol ikon, kontras warna yang cukup, umpan balik yang aman untuk buta warna (jangan hanya mengandalkan warna, tambahkan ikon atau teks), dan hormati pengaturan prefers-reduced-motion.',
  })],
];

// Preset text fields are { en, id } pairs so they load in the current interface language.
const PRESETS = {
  sahabatQuiz: {
    title: { en: 'Sahabat Rasul Challenge', id: 'Tantangan Sahabat Rasul' }, genre: 'quiz',
    theme: { en: 'companions of Prophet Muhammad ﷺ — their names, stories, and contributions to Islam', id: 'sahabat Nabi Muhammad ﷺ — nama, kisah, dan kontribusi mereka terhadap Islam' },
    concept: { en: 'Answer questions about the Sahabat to build a golden mosque. Each correct answer adds a pillar. Wrong answers crack a wall. Learn about Abu Bakr, Umar, Uthman, Ali, Khadijah, Bilal, and more.', id: 'Jawab pertanyaan tentang Sahabat untuk membangun masjid emas. Setiap jawaban benar menambah pilar. Jawaban salah meretakkan dinding. Pelajari Abu Bakar, Umar, Utsman, Ali, Khadijah, Bilal, dan lainnya.' },
    audience: 'kids', purpose: 'education',
    objective: { en: 'Students can identify 10+ Sahabat Rasul and describe their key roles in early Islamic history', id: 'Siswa dapat mengenal 10+ Sahabat Rasul dan menjelaskan peran utama mereka dalam sejarah awal Islam' },
    levels: 3, lives: 3, session: '5', difficulty: 'easy', style: 'cartoon', palette: 'ocean', lang: 'bi', mood: 'cheerful',
    features: ['score', 'highscore', 'timer', 'lives', 'levels', 'difficultySelect', 'pause', 'tutorial', 'sfx', 'juice', 'avatar', 'achievements', 'mobile', 'a11y'],
  },
  arabicVocab: {
    title: { en: 'Arabic Word Explorer', id: 'Penjelajah Kata Arab' }, genre: 'quiz',
    theme: { en: 'Arabic vocabulary — everyday words, Quranic terms, greetings, numbers, colors, body parts, and family words with Arabic script (حروف عربية)', id: 'kosa kata bahasa Arab — kata sehari-hari, istilah Al-Quran, sapaan, angka, warna, anggota tubuh, dan kata keluarga dengan tulisan Arab (حروف عربية)' },
    concept: { en: 'See an Arabic word in its script, hear it read aloud, then pick the correct meaning. Earn stars to unlock new word categories. Includes Arabic writing display for every word.', id: 'Lihat kata Arab dalam tulisan aslinya, dengarkan pengucapannya, lalu pilih arti yang benar. Kumpulkan bintang untuk membuka kategori kata baru. Menampilkan tulisan Arab untuk setiap kata.' },
    audience: 'preteens', purpose: 'education',
    objective: { en: 'Learners can recognize and translate 50+ basic Arabic words including their written Arabic form', id: 'Siswa dapat mengenali dan menerjemahkan 50+ kata dasar bahasa Arab termasuk bentuk tulisan Arabnya' },
    levels: 5, lives: 3, session: '5', difficulty: 'easy', style: 'flat', palette: 'sunset', lang: 'ar', mood: 'calm',
    features: ['score', 'highscore', 'timer', 'lives', 'levels', 'difficultySelect', 'pause', 'tutorial', 'sfx', 'juice', 'achievements', 'hints', 'mobile', 'a11y'],
  },
  heroAdventure: {
    title: { en: 'Muslim Heroes Journey', id: 'Perjalanan Pahlawan Muslim' }, genre: 'adventure',
    theme: { en: 'exploring the lives of great Muslim heroes from the era of Rasulullah ﷺ — Khalid ibn al-Walid, Salahuddin, Hamzah ibn Abdul-Muttalib, Musab ibn Umair, and others', id: 'menjelajahi kehidupan pahlawan Muslim hebat dari era Rasulullah ﷺ — Khalid bin Walid, Salahuddin, Hamzah bin Abdul Muthalib, Mush\'ab bin Umair, dan lainnya' },
    concept: { en: 'Travel through historical Islamic locations, meet Muslim heroes, answer questions about their stories, and collect wisdom scrolls. Each hero teaches a virtue: bravery, justice, generosity, or knowledge.', id: 'Berpetualang ke lokasi bersejarah Islam, bertemu pahlawan Muslim, jawab pertanyaan tentang kisah mereka, dan kumpulkan gulungan hikmah. Setiap pahlawan mengajarkan satu kebajikan: keberanian, keadilan, kedermawanan, atau ilmu.' },
    audience: 'preteens', purpose: 'education',
    objective: { en: 'Students can narrate the key stories of 8+ Muslim heroes and the virtues they represent', id: 'Siswa dapat menceritakan kisah utama 8+ pahlawan Muslim dan kebajikan yang mereka wakili' },
    levels: 5, lives: 3, session: '10', difficulty: 'normal', style: 'fantasy', palette: 'earth', lang: 'bi', mood: 'mysterious',
    features: ['score', 'levels', 'pause', 'tutorial', 'sfx', 'music', 'juice', 'achievements', 'hints', 'settings', 'mobile', 'a11y'],
  },
  hijaiyahMemory: {
    title: { en: 'Hijaiyah Match', id: 'Cocokkan Hijaiyah' }, genre: 'memory',
    theme: { en: 'Arabic letters (Hijaiyah / huruf hijaiyah) — match each letter with its name and sound: alif (ا), ba (ب), ta (ت), tsa (ث), jim (ج), and all 28 letters', id: 'huruf hijaiyah — cocokkan setiap huruf dengan nama dan bunyinya: alif (ا), ba (ب), ta (ت), tsa (ث), jim (ج), dan semua 28 huruf' },
    concept: { en: 'Flip cards to match Arabic letters with their romanized names. The cards show beautiful Arabic calligraphy. Progress from basic letters to letters with harakat (fathah, kasrah, dhammah).', id: 'Balik kartu untuk mencocokkan huruf Arab dengan nama latinnya. Kartu menampilkan kaligrafi Arab yang indah. Berlanjut dari huruf dasar ke huruf dengan harakat (fathah, kasrah, dhammah).' },
    audience: 'kids', purpose: 'education',
    objective: { en: 'Learners can identify all 28 Hijaiyah letters and their basic sounds', id: 'Siswa dapat mengenal semua 28 huruf Hijaiyah dan bunyi dasarnya' },
    levels: 3, lives: 3, session: '5', difficulty: 'easy', style: 'flat', palette: 'candy', lang: 'ar', mood: 'cheerful',
    features: ['score', 'highscore', 'timer', 'levels', 'pause', 'tutorial', 'sfx', 'juice', 'achievements', 'mobile', 'a11y'],
  },
  rukunIman: {
    title: { en: 'Pillars of Faith Challenge', id: 'Tantangan Rukun Iman' }, genre: 'quiz',
    theme: { en: 'the 6 Pillars of Faith (Rukun Iman) — belief in Allah, Angels (Malaikat), Holy Books (Kitab), Prophets (Rasul), the Day of Judgment (Hari Akhir), and Qadar (destiny)', id: 'Rukun Iman yang 6 — iman kepada Allah, Malaikat, Kitab-kitab, Rasul-rasul, Hari Akhir, dan Qada & Qadar' },
    concept: { en: 'Answer questions about each pillar of faith. Each correct answer lights up a star on a crescent. Complete all 6 pillars to unlock a golden crescent and moon. Questions range from naming the pillars to understanding their meaning and evidence from Quran and Hadith.', id: 'Jawab pertanyaan tentang setiap rukun iman. Setiap jawaban benar menyalakan bintang di bulan sabit. Selesaikan 6 rukun untuk membuka bulan sabit emas. Pertanyaan mulai dari menyebut rukun hingga memahami makna dan dalil dari Al-Quran dan Hadis.' },
    audience: 'kids', purpose: 'education',
    objective: { en: 'Students can name and explain all 6 Rukun Iman with evidence from Quran and Hadith', id: 'Siswa dapat menyebutkan dan menjelaskan 6 Rukun Iman beserta dalil dari Al-Quran dan Hadis' },
    levels: 3, lives: 3, session: '5', difficulty: 'easy', style: 'cartoon', palette: 'ocean', lang: 'bi', mood: 'cheerful',
    features: ['score', 'highscore', 'timer', 'lives', 'levels', 'difficultySelect', 'pause', 'tutorial', 'sfx', 'juice', 'achievements', 'hints', 'mobile', 'a11y'],
  },
  rukunIslam: {
    title: { en: 'Five Pillars of Islam', id: 'Lima Rukun Islam' }, genre: 'quiz',
    theme: { en: 'the 5 Pillars of Islam (Rukun Islam) — Syahadah (declaration of faith), Salah (prayer), Zakat (charity), Sawm (fasting in Ramadan), and Hajj (pilgrimage to Makkah)', id: 'Rukun Islam yang 5 — Syahadat, Shalat, Zakat, Puasa Ramadhan, dan Haji ke Makkah' },
    concept: { en: 'Build a beautiful mosque brick by brick! Each correct answer about the 5 pillars adds a brick. Learn about Syahadah wording, the 5 daily prayers and their times, who receives Zakat, Ramadan fasting rules, and Hajj rituals. Includes Arabic text for key terms and du\'a.', id: 'Bangun masjid yang indah bata demi bata! Setiap jawaban benar tentang 5 rukun menambah bata. Pelajari lafaz Syahadat, 5 waktu shalat, siapa penerima Zakat, aturan puasa Ramadhan, dan ritual Haji. Menampilkan teks Arab untuk istilah kunci dan doa.' },
    audience: 'kids', purpose: 'education',
    objective: { en: 'Students can explain the 5 Pillars of Islam, their requirements, and practical application in daily Muslim life', id: 'Siswa dapat menjelaskan 5 Rukun Islam, syarat-syaratnya, dan penerapan praktis dalam kehidupan Muslim sehari-hari' },
    levels: 5, lives: 3, session: '5', difficulty: 'easy', style: 'flat', palette: 'sunset', lang: 'bi', mood: 'cheerful',
    features: ['score', 'highscore', 'timer', 'lives', 'levels', 'difficultySelect', 'pause', 'tutorial', 'sfx', 'juice', 'avatar', 'achievements', 'hints', 'mobile', 'a11y'],
  },
  tebakAyat: {
    title: { en: 'Guess the Ayah', id: 'Tebak Ayat Al-Quran' }, genre: 'quiz',
    theme: { en: 'Quranic verses (ayat) — guess which Surah an ayah belongs to, complete missing words in a verse, or match Arabic text with its translation. Features verses from Al-Fatihah, Al-Baqarah, Al-Ikhlas, An-Nas, Al-Falaq, Yasin, Ar-Rahman, and other popular Surahs.', id: 'ayat-ayat Al-Quran — tebak dari Surah mana sebuah ayat, lengkapi kata yang hilang, atau cocokkan teks Arab dengan terjemahannya. Menampilkan ayat dari Al-Fatihah, Al-Baqarah, Al-Ikhlas, An-Nas, Al-Falaq, Yasin, Ar-Rahman, dan Surah populer lainnya.' },
    concept: { en: 'See a Quranic verse displayed in beautiful Arabic calligraphy. Choose the correct Surah name, or fill in the missing word. Harder levels show only the translation and you must identify the Surah. Earn crescent moons for streaks. Learn the meaning and context (asbabun nuzul) of each verse.', id: 'Lihat ayat Al-Quran ditampilkan dalam kaligrafi Arab yang indah. Pilih nama Surah yang benar, atau isi kata yang hilang. Level lebih sulit hanya menampilkan terjemahan dan kamu harus menebak Surahnya. Kumpulkan bulan sabit untuk streak. Pelajari makna dan konteks (asbabun nuzul) setiap ayat.' },
    audience: 'preteens', purpose: 'education',
    objective: { en: 'Students can identify 30+ Quranic verses, name their Surah, and explain their meaning', id: 'Siswa dapat mengenali 30+ ayat Al-Quran, menyebut nama Surahnya, dan menjelaskan maknanya' },
    levels: 5, lives: 3, session: '10', difficulty: 'normal', style: 'minimal', palette: 'earth', lang: 'ar', mood: 'calm',
    features: ['score', 'highscore', 'timer', 'lives', 'levels', 'difficultySelect', 'pause', 'tutorial', 'sfx', 'juice', 'achievements', 'hints', 'mobile', 'a11y'],
  },
  quranExplorer: {
    title: { en: 'Quran Knowledge Explorer', id: 'Jelajah Pengetahuan Al-Quran' }, genre: 'adventure',
    theme: { en: 'exploring the Quran — learn about the 114 Surahs, their themes, Makki vs Madani classification, stories of the Prophets mentioned in the Quran (Nuh, Ibrahim, Musa, Isa, Muhammad ﷺ), and key Islamic values: justice (العدل), mercy (الرحمة), patience (الصبر), gratitude (الشكر), honesty (الصدق)', id: 'menjelajahi Al-Quran — pelajari 114 Surah, tema-temanya, klasifikasi Makkiyah vs Madaniyah, kisah para Nabi dalam Al-Quran (Nuh, Ibrahim, Musa, Isa, Muhammad ﷺ), dan nilai-nilai Islam utama: keadilan (العدل), kasih sayang (الرحمة), kesabaran (الصبر), rasa syukur (الشكر), kejujuran (الصدق)' },
    concept: { en: 'Journey through a beautiful library of the Quran. Each room represents a Juz or theme. Solve puzzles about Surah names, prophets\' stories, and Islamic values to unlock the next room. Collect wisdom gems that reveal key verses with Arabic text and translation.', id: 'Berpetualang melalui perpustakaan Al-Quran yang indah. Setiap ruangan mewakili satu Juz atau tema. Pecahkan teka-teki tentang nama Surah, kisah para Nabi, dan nilai-nilai Islam untuk membuka ruangan berikutnya. Kumpulkan permata hikmah yang menampilkan ayat-ayat penting beserta teks Arab dan terjemahannya.' },
    audience: 'preteens', purpose: 'education',
    objective: { en: 'Students gain comprehensive knowledge of Quran structure, prophet stories, and core Islamic values with supporting verses', id: 'Siswa memperoleh pengetahuan komprehensif tentang struktur Al-Quran, kisah para Nabi, dan nilai-nilai Islam utama beserta ayat pendukungnya' },
    levels: 5, lives: 3, session: '10', difficulty: 'normal', style: 'fantasy', palette: 'earth', lang: 'bi', mood: 'mysterious',
    features: ['score', 'levels', 'pause', 'tutorial', 'sfx', 'music', 'juice', 'achievements', 'hints', 'settings', 'mobile', 'a11y'],
  },
  bacaanSholat: {
    title: { en: 'Prayer Recitation Master', id: 'Hafalan Bacaan Sholat' }, genre: 'quiz',
    theme: { en: 'memorizing prayer recitations (bacaan sholat) — Niat, Takbiratul Ihram (الله أكبر), Al-Fatihah, Ruku\' recitation (سبحان ربي العظيم), I\'tidal (سمع الله لمن حمده), Sujud recitation (سبحان ربي الأعلى), sitting between two sujud, Tasyahhud/Tahiyyat, Shalawat Ibrahimiyyah, and Salam', id: 'hafalan bacaan sholat — Niat, Takbiratul Ihram (الله أكبر), Al-Fatihah, bacaan Ruku\' (سبحان ربي العظيم), I\'tidal (سمع الله لمن حمده), bacaan Sujud (سبحان ربي الأعلى), duduk antara dua sujud, Tasyahhud/Tahiyyat, Shalawat Ibrahimiyyah, dan Salam' },
    concept: { en: 'Follow along a visual prayer sequence. At each position (standing, bowing, prostrating, sitting), fill in the missing words of the recitation in Arabic. Includes transliteration and meaning. Progress from short prayers (Subuh) to longer ones (Dhuhur, Ashar, Maghrib, Isya).', id: 'Ikuti urutan sholat secara visual. Di setiap posisi (berdiri, ruku\', sujud, duduk), lengkapi kata yang hilang dari bacaan dalam bahasa Arab. Dilengkapi transliterasi dan arti. Berlanjut dari sholat pendek (Subuh) ke yang lebih panjang (Dhuhur, Ashar, Maghrib, Isya).' },
    audience: 'kids', purpose: 'education',
    objective: { en: 'Students can recite all prayer recitations in correct order with proper Arabic pronunciation', id: 'Siswa dapat membaca semua bacaan sholat dalam urutan yang benar dengan pengucapan Arab yang tepat' },
    levels: 5, lives: 3, session: '5', difficulty: 'easy', style: 'flat', palette: 'ocean', lang: 'ar', mood: 'calm',
    features: ['score', 'highscore', 'lives', 'levels', 'difficultySelect', 'pause', 'tutorial', 'sfx', 'juice', 'achievements', 'hints', 'mobile', 'a11y'],
  },
  doaHarian: {
    title: { en: 'Daily Du\'a Champion', id: 'Juara Doa Harian' }, genre: 'memory',
    theme: { en: 'daily Islamic du\'a (prayers/supplications) — du\'a before eating (بسم الله), after eating (الحمد لله), entering the mosque, leaving the mosque, before sleeping, waking up, entering the bathroom, traveling, for parents, when it rains, when looking in a mirror, and more — all in Arabic with meaning', id: 'doa harian Islam — doa sebelum makan (بسم الله), sesudah makan (الحمد لله), masuk masjid, keluar masjid, sebelum tidur, bangun tidur, masuk kamar mandi, bepergian, untuk orang tua, saat hujan, saat bercermin, dan lainnya — semua dalam bahasa Arab beserta artinya' },
    concept: { en: 'Match each daily activity with its correct du\'a. Cards show the activity on one side and the Arabic du\'a text on the other. Learn when and why to say each du\'a. Progress from simple (Bismillah, Alhamdulillah) to longer supplications.', id: 'Cocokkan setiap aktivitas harian dengan doa yang tepat. Kartu menampilkan aktivitas di satu sisi dan teks doa Arab di sisi lain. Pelajari kapan dan mengapa membaca setiap doa. Berlanjut dari yang sederhana (Bismillah, Alhamdulillah) ke doa yang lebih panjang.' },
    audience: 'kids', purpose: 'education',
    objective: { en: 'Students can recite 15+ daily du\'a in Arabic and know when to use each one', id: 'Siswa dapat membaca 15+ doa harian dalam bahasa Arab dan tahu kapan menggunakan masing-masing' },
    levels: 3, lives: 3, session: '5', difficulty: 'easy', style: 'cartoon', palette: 'candy', lang: 'ar', mood: 'cheerful',
    features: ['score', 'highscore', 'timer', 'levels', 'pause', 'tutorial', 'sfx', 'juice', 'achievements', 'mobile', 'a11y'],
  },
};

const FOLLOWUPS = {
  en: [
    'Add more Islamic content: include additional Sahabat, Quranic vocabulary, or Muslim heroes with their stories.',
    'Display Arabic text (حروف عربية) beautifully with proper right-to-left rendering and large calligraphy-style fonts.',
    'Add 3 more levels that introduce new topics (e.g. Surahs, Hadith themes, or Arabic grammar basics).',
    'Improve the mobile experience: test at 375 px width and make the touch controls bigger and easier to use.',
    'Add a short animated intro with Islamic geometric patterns on the title screen and a "MasyaAllah!" celebration screen for winning.',
    'Fix any bugs you can find, and make sure there are no errors in the browser console.',
  ],
  id: [
    'Tambahkan lebih banyak konten Islam: sertakan Sahabat tambahan, kosa kata Al-Quran, atau pahlawan Muslim beserta kisahnya.',
    'Tampilkan teks Arab (حروف عربية) dengan indah menggunakan render kanan-ke-kiri yang tepat dan font kaligrafi besar.',
    'Tambahkan 3 level lagi yang memperkenalkan topik baru (mis. Surah, tema Hadis, atau dasar tata bahasa Arab).',
    'Perbaiki pengalaman di HP: uji di lebar 375 px dan buat kontrol sentuh lebih besar dan mudah dipakai.',
    'Tambahkan intro animasi singkat dengan pola geometris Islam di layar judul dan layar perayaan "MasyaAllah!" saat menang.',
    'Perbaiki semua bug yang kamu temukan, dan pastikan tidak ada error di konsol browser.',
  ],
};

// ---------- Helpers ----------

const $ = (sel) => document.querySelector(sel);
const form = $('#builder');

let uiLang = 'en';  // interface language
let L = 'en';       // language used while building the prompt

/** Pick the current prompt-language version of an { en, id } pair. */
function T(pair) { return pair[L]; }
/** Count with a noun: English pluralises, Indonesian does not. */
function n(lang, count, one, many = one + 's') {
  return lang === 'en' ? `${count} ${count === 1 ? one : many}` : `${count} ${one}`;
}
function clampInt(v, min, max, fallback) {
  const x = parseInt(v, 10);
  return Number.isFinite(x) ? Math.min(max, Math.max(min, x)) : fallback;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
}
function numbered(items) { return items.map((t, i) => `   ${i + 1}. ${t}`).join('\n'); }
function bullets(items) { return items.map((t) => `   - ${t}`).join('\n'); }
function sub(items) { return items.map((t) => `      - ${t}`).join('\n'); }

// ---------- Interface ----------

function fillSelect(sel, entries, fallback) {
  const prev = sel.value || fallback;
  sel.innerHTML = entries.map(([v, label]) => `<option value="${v}">${escapeHtml(label)}</option>`).join('');
  sel.value = entries.some(([v]) => v === prev) ? prev : fallback;
}

/** Apply the interface language to all static text, selects and feature labels. */
function applyUiLang(lang) {
  uiLang = lang;
  const s = UI[lang];
  document.documentElement.lang = lang;
  document.querySelectorAll('[data-i18n]').forEach((el) => { el.textContent = s[el.dataset.i18n]; });
  document.querySelectorAll('[data-i18n-html]').forEach((el) => { el.innerHTML = s[el.dataset.i18nHtml]; });
  document.querySelectorAll('[data-i18n-ph]').forEach((el) => { el.placeholder = s[el.dataset.i18nPh]; });
  document.querySelectorAll('[data-i18n-aria]').forEach((el) => { el.setAttribute('aria-label', s[el.dataset.i18nAria]); });
  document.querySelectorAll('[data-ui-lang]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.uiLang === lang)));

  fillSelect($('#genre'), Object.entries(GENRES).map(([id, g]) => [id, g.label[lang]]), 'quiz');
  document.querySelectorAll('select[data-options]').forEach((sel) => {
    const data = SELECT_DATA[sel.dataset.options];
    fillSelect(sel, Object.entries(data).map(([id, o]) => [id, o.label[lang]]), SELECT_DEFAULTS[sel.name]);
  });

  // Feature checkboxes: keep checked state across language switches.
  const box = $('#features');
  const checked = box.children.length ? new Set(FEATURES.filter(([id]) => form.elements['f_' + id].checked).map(([id]) => id)) : null;
  box.innerHTML = FEATURES.map(([id, label, on]) =>
    `<label class="check"><input type="checkbox" name="f_${id}" ${(checked ? checked.has(id) : on) ? 'checked' : ''}><span>${escapeHtml(label[lang])}</span></label>`
  ).join('');

  $('#legend').innerHTML = s.legend.map((t, i) => `<span class="${'star'[i]}">${t}</span>`).join('');
  $('#followups').innerHTML = FOLLOWUPS[lang].map((f) => `<li title="${escapeHtml(s.followHint)}">${escapeHtml(f)}</li>`).join('');
}

function readConfig() {
  const d = new FormData(form);
  const get = (k) => (d.get(k) || '').toString().trim();
  const palette = get('palette') || 'auto';
  const colors = palette === 'custom' ? [get('c1'), get('c2'), get('c3')] : PALETTES[palette].colors;
  return {
    promptLang: get('promptLang') === 'id' ? 'id' : 'en',
    title: get('title'),
    genre: GENRES[get('genre')] ? get('genre') : 'quiz',
    theme: get('theme'),
    concept: get('concept'),
    audience: get('audience') || 'all',
    purpose: get('purpose') || 'fun',
    objective: get('objective'),
    levels: clampInt(get('levels'), 1, 20, 3),
    lives: clampInt(get('lives'), 1, 10, 3),
    session: get('session') || '5',
    difficulty: get('difficulty') || 'normal',
    style: get('style') || 'flat',
    palette,
    colors,
    lang: get('lang') || 'en',
    mood: get('mood') || 'cheerful',
    format: get('format'),
    comments: d.has('comments'),
    localhost: d.has('localhost'),
    nextIdeas: d.has('nextIdeas'),
    extra: get('extra'),
    useContext: d.has('useContext'),
    uploadedContext: uploadedContextText,
    features: FEATURES.filter(([id]) => d.has('f_' + id)).map(([id]) => id),
  };
}

// ---------- Prompt builder (STAR) ----------

function buildPrompt(cfg) {
  L = cfg.promptLang;
  const c = { ...cfg, theme: cfg.theme || T({ en: 'a theme of your choice', id: 'tema pilihanmu sendiri' }) };
  const g = GENRES[c.genre];
  const aud = AUDIENCES[c.audience];
  const has = (id) => c.features.includes(id);
  const isEdu = c.purpose === 'education' || c.purpose === 'training';
  const single = c.format === 'single';
  const genreName = T(g.label);
  const mood = T(MOODS[c.mood].text);
  const who = T(aud.who);
  const diffName = DIFFICULTY[c.difficulty][L];

  // --- SITUATION ---
  const titleText = c.title ? `"${c.title}"` : T({ en: 'a catchy title you invent', id: 'judul menarik yang kamu buat sendiri' });
  const situation = [
    T({
      en: `I want to create an interactive browser game called ${titleText}. It is a ${genreName} game about ${c.theme}.`,
      id: `Saya ingin membuat game interaktif berbasis browser dengan judul ${titleText}. Ini adalah game ${genreName} tentang ${c.theme}.`,
    }),
    c.concept ? T({ en: 'Core idea: ', id: 'Ide utama: ' }) + c.concept : '',
    '',
    bullets([
      T({ en: 'Players: ', id: 'Pemain: ' }) + `${who}. ${T(aud.ux)}`,
      T({ en: 'Purpose: ', id: 'Tujuan: ' }) + T(PURPOSES[c.purpose].text) + '.',
      isEdu && c.objective ? T({ en: 'Learning objective: ', id: 'Tujuan pembelajaran: ' }) + c.objective + '.' : '',
      T({ en: `A typical play session lasts ${T(SESSIONS[c.session].text)}.`, id: `Satu sesi bermain berlangsung ${T(SESSIONS[c.session].text)}.` }),
      T({ en: 'Mood: ', id: 'Suasana: ' }) + mood + '.',
      T({ en: `All in-game text must be in ${T(GAME_LANGS[c.lang].text)}.`, id: `Semua teks di dalam game harus dalam ${T(GAME_LANGS[c.lang].text)}.` }),
      T({
        en: 'It will be previewed in Gemini Canvas first and then run on a local web server (localhost) on desktop and mobile browsers.',
        id: 'Game ini akan dipratinjau di Gemini Canvas terlebih dahulu, lalu dijalankan di server web lokal (localhost) pada browser desktop dan HP.',
      }),
      T({
        en: 'Constraints: no build tools, no frameworks, no backend and no external image or audio files. Everything is drawn with HTML, CSS, SVG, canvas or emoji, and sound is generated with code. Google Fonts are the only allowed external resource.',
        id: 'Batasan: tanpa build tool, tanpa framework, tanpa backend, dan tanpa file gambar atau audio dari luar. Semua digambar dengan HTML, CSS, SVG, canvas, atau emoji, dan suara dibuat lewat kode. Google Fonts adalah satu-satunya sumber eksternal yang boleh dipakai.',
      }),
    ].filter(Boolean)),
  ].filter((line, i, a) => !(line === '' && a[i - 1] === '')).join('\n');

  // --- TASK ---
  const deliverable = single
    ? T({
      en: 'one complete, self-contained index.html file that contains all the HTML, CSS (in a <style> tag) and JavaScript (in a <script> tag)',
      id: 'satu file index.html yang lengkap dan mandiri, berisi semua HTML, CSS (di dalam tag <style>), dan JavaScript (di dalam tag <script>)',
    })
    : T({
      en: 'a small project with three files: index.html, style.css and script.js, which index.html links to with relative paths',
      id: 'proyek kecil dengan tiga file: index.html, style.css, dan script.js, yang ditautkan dari index.html dengan path relatif',
    });
  const task = [
    T({
      en: `Act as a senior HTML5 game developer and UI/UX designer. Build a complete, polished and fully playable ${genreName} game as ${deliverable}.`,
      id: `Bertindaklah sebagai senior HTML5 game developer dan desainer UI/UX. Buat game ${genreName} yang lengkap, rapi, dan bisa dimainkan sepenuhnya dalam bentuk ${deliverable}.`,
    }),
    T({ en: `Build the game with ${T(g.tech)}.`, id: `Bangun game ini dengan ${T(g.tech)}.` }),
    T({
      en: 'The game must be finished and working, not a prototype: no placeholders, no "TODO" comments and no "add your content here" parts.',
      id: 'Game harus selesai dan berfungsi, bukan prototipe: tanpa placeholder, tanpa komentar "TODO", dan tanpa bagian "isi kontenmu di sini".',
    }),
  ].join('\n');

  // --- ACTION ---
  const screens = [
    T({ en: 'Start screen with the game title, a short tagline and a Play button', id: 'Layar awal dengan judul game, slogan singkat, dan tombol Main' }),
    has('tutorial') && T({ en: '"How to Play" screen', id: 'layar "Cara Bermain"' }),
    has('avatar') && T({ en: 'character selection screen', id: 'layar pilih karakter' }),
    has('settings') && T({ en: 'settings screen', id: 'layar pengaturan' }),
    T({ en: 'game screen with a clear HUD', id: 'layar permainan dengan HUD yang jelas' }),
    has('pause') && T({ en: 'pause overlay', id: 'tampilan jeda' }),
    c.levels > 1 && has('levels') && T({ en: 'level-complete screen', id: 'layar level selesai' }),
    T({ en: 'game-over / victory screen with the final score and Play Again and Main Menu buttons', id: 'layar game over / menang dengan skor akhir serta tombol Main Lagi dan Menu Utama' }),
    has('leaderboard') && T({ en: 'leaderboard screen', id: 'layar papan peringkat' }),
    has('achievements') && T({ en: 'achievements screen', id: 'layar pencapaian' }),
  ].filter(Boolean);

  const hud = [
    has('score') && T({ en: 'score', id: 'skor' }),
    has('lives') && T({ en: 'lives', id: 'nyawa' }),
    has('timer') && T({ en: 'timer', id: 'timer' }),
    has('levels') && c.levels > 1 && T({ en: 'current level', id: 'level saat ini' }),
    has('hints') && T({ en: 'hints left', id: 'sisa petunjuk' }),
    has('sfx') && T({ en: 'mute button', id: 'tombol bisu' }),
    has('pause') && T({ en: 'pause button', id: 'tombol jeda' }),
  ].filter(Boolean);

  const featureLines = FEATURES.filter(([id]) => has(id)).map(([, , , fn]) => fn(c));

  const paletteLine = c.colors.length
    ? T({
      en: `Colour palette: ${c.colors.join(', ')} (primary, accent, background${c.colors.length > 3 ? ', highlight' : ''}). Define these as CSS custom properties on :root and use them everywhere.`,
      id: `Palet warna: ${c.colors.join(', ')} (utama, aksen, latar${c.colors.length > 3 ? ', sorotan' : ''}). Definisikan sebagai CSS custom properties di :root dan gunakan di semua bagian.`,
    })
    : T({
      en: 'Choose a cohesive colour palette that fits the theme and mood, and define the colours as CSS custom properties on :root.',
      id: 'Pilih palet warna yang serasi dengan tema dan suasana, dan definisikan warnanya sebagai CSS custom properties di :root.',
    });

  const codeRules = [
    T({ en: 'Plain ("vanilla") JavaScript (ES6+) only; no libraries.', id: 'Hanya JavaScript murni ("vanilla", ES6+); tanpa library.' }),
    T({
      en: 'Organise the code into clear sections: CONFIG (all tunable numbers in one object), STATE, INPUT, UPDATE, RENDER, UI/SCREENS, AUDIO and STORAGE.',
      id: 'Susun kode dalam bagian-bagian yang jelas: CONFIG (semua angka yang bisa diatur dalam satu objek), STATE, INPUT, UPDATE, RENDER, UI/SCREENS, AUDIO, dan STORAGE.',
    }),
    T({
      en: 'Use a single state machine for screens (menu, playing, paused, levelComplete, gameOver) so there are no overlapping timers or duplicate event listeners when the player restarts.',
      id: 'Gunakan satu state machine untuk layar (menu, playing, paused, levelComplete, gameOver) agar tidak ada timer yang tumpang tindih atau event listener ganda saat pemain mengulang.',
    }),
    T({ en: 'Restarting or playing again must fully reset the game state.', id: 'Mengulang atau main lagi harus mengembalikan seluruh state game ke awal.' }),
    T({
      en: 'Handle window resizing and high-DPI (devicePixelRatio) screens so graphics stay sharp.',
      id: 'Tangani perubahan ukuran jendela dan layar high-DPI (devicePixelRatio) agar grafis tetap tajam.',
    }),
    c.comments ? T({
      en: 'Add clear comments that explain each section and any non-obvious logic, so a beginner can change the game.',
      id: 'Tambahkan komentar yang jelas untuk menjelaskan setiap bagian dan logika yang tidak langsung terlihat, agar pemula pun bisa mengubah game-nya.',
    }) : '',
    T({ en: 'There must be no errors or warnings in the browser console.', id: 'Tidak boleh ada error atau peringatan di konsol browser.' }),
  ].filter(Boolean);

  const actionSteps = [
    T({ en: 'Screens and flow. Build these screens with smooth transitions: ', id: 'Layar dan alur. Buat layar-layar berikut dengan transisi yang halus: ' }) + screens.join(' → ') + '.',
    T({ en: 'Core gameplay.', id: 'Gameplay inti.' }) + '\n' + sub(T(g.mechanics)),
    T({ en: 'Controls. ', id: 'Kontrol. ' }) + T(g.controls),
    T({ en: 'Content. ', id: 'Konten. ' }) + g.content[L](c),
    hud.length ? T({
      en: `HUD. Always show: ${hud.join(', ')}. Keep it readable and out of the way of the gameplay.`,
      id: `HUD. Selalu tampilkan: ${hud.join(', ')}. Buat mudah dibaca dan tidak menghalangi permainan.`,
    }) : '',
    featureLines.length ? T({ en: 'Features.', id: 'Fitur.' }) + '\n' + sub(featureLines) : '',
    T({
      en: `Visual design. Use a ${T(STYLES[c.style].text)}. ${paletteLine} Give the game a polished feel that matches the "${mood}" mood, with a consistent type scale and spacing.`,
      id: `Desain visual. Gunakan ${T(STYLES[c.style].text)}. ${paletteLine} Beri kesan rapi dan profesional yang sesuai dengan suasana "${mood}", dengan ukuran huruf dan jarak yang konsisten.`,
    }),
    T({
      en: `Difficulty balance. The ${diffName} setting should feel right for ${who}: the first minute teaches the game, and the challenge grows steadily after that.`,
      id: `Keseimbangan tingkat kesulitan. Pengaturan ${diffName} harus terasa pas untuk ${who}: menit pertama mengajarkan cara bermain, lalu tantangannya naik secara bertahap.`,
    }),
    T({ en: 'Code quality.', id: 'Kualitas kode.' }) + '\n' + sub(codeRules),
    c.extra ? T({ en: 'Extra requirements. ', id: 'Permintaan tambahan. ' }) + c.extra : '',
    (c.useContext && c.uploadedContext) ? T({ en: 'Reference material (from uploaded documents). Use the following content as the knowledge base for generating questions, quiz items, game content, or learning material:\n\n', id: 'Materi referensi (dari dokumen yang diunggah). Gunakan konten berikut sebagai basis pengetahuan untuk membuat pertanyaan, item kuis, konten game, atau materi pembelajaran:\n\n' }) + '```\n' + c.uploadedContext.slice(0, 12000) + '\n```' : '',
  ].filter(Boolean);

  const action = T({ en: 'Follow these steps:', id: 'Ikuti langkah-langkah berikut:' }) + '\n' + numbered(actionSteps);

  // --- RESULT ---
  const criteria = [
    T({
      en: 'The game loads straight away in the Gemini Canvas preview and can be played from the start screen to the end without errors.',
      id: 'Game langsung termuat di pratinjau Gemini Canvas dan bisa dimainkan dari layar awal sampai akhir tanpa error.',
    }),
    T({ en: 'Every selected feature above is implemented and works.', id: 'Semua fitur yang dipilih di atas sudah dibuat dan berfungsi.' }),
    T({
      en: 'It works with mouse, keyboard and touch, and looks good on both phone and desktop screens.',
      id: 'Bisa dimainkan dengan mouse, keyboard, dan sentuhan, serta tampil bagus di layar HP maupun desktop.',
    }),
    has('highscore') || has('leaderboard') || has('achievements') ? T({
      en: 'Saved progress survives a page reload, and the game still runs if localStorage is unavailable.',
      id: 'Progres yang tersimpan tetap ada setelah halaman dimuat ulang, dan game tetap berjalan jika localStorage tidak tersedia.',
    }) : '',
    isEdu ? T({
      en: 'All educational content is accurate and suitable for the target players.',
      id: 'Semua materi pembelajaran akurat dan sesuai untuk target pemain.',
    }) : '',
    T({
      en: 'A first-time player understands what to do within 10 seconds without outside help.',
      id: 'Pemain baru langsung paham apa yang harus dilakukan dalam 10 detik tanpa bantuan.',
    }),
  ].filter(Boolean);

  const outputLines = [
    single
      ? T({ en: 'Return the complete index.html in a single code block, so it renders in Canvas immediately.', id: 'Berikan index.html yang lengkap dalam satu blok kode, agar langsung tampil di Canvas.' })
      : T({
        en: 'Return each file (index.html, style.css, script.js) in its own code block with the filename as a heading, followed by the folder structure.',
        id: 'Berikan setiap file (index.html, style.css, script.js) dalam blok kode masing-masing dengan nama file sebagai judul, lalu tampilkan struktur foldernya.',
      }),
    T({ en: 'Do not leave out or shorten any code.', id: 'Jangan menghilangkan atau menyingkat bagian kode mana pun.' }),
    c.localhost ? T({
      en: `After the code, add a short "Run it on localhost" section: save the ${single ? 'file as index.html' : 'files'} in a new folder, open a terminal in that folder, run \`python3 -m http.server 8000\` (or \`npx serve\`), and open http://localhost:8000 in a browser.`,
      id: `Setelah kode, tambahkan bagian singkat "Menjalankan di localhost": simpan ${single ? 'file sebagai index.html' : 'file-file tersebut'} di folder baru, buka terminal di folder itu, jalankan \`python3 -m http.server 8000\` (atau \`npx serve\`), lalu buka http://localhost:8000 di browser.`,
    }) : '',
    T({ en: 'Add a 3–5 line "How to play" summary for the player.', id: 'Tambahkan ringkasan "Cara bermain" sepanjang 3–5 baris untuk pemain.' }),
    c.nextIdeas ? T({ en: 'End with 3 specific ideas for improving the game in a next version.', id: 'Akhiri dengan 3 ide spesifik untuk mengembangkan game di versi berikutnya.' }) : '',
  ].filter(Boolean);

  const result = [
    T({ en: 'Definition of done:', id: 'Kriteria selesai:' }),
    bullets(criteria),
    '',
    T({ en: 'Output format:', id: 'Format hasil:' }),
    bullets(outputLines),
  ].join('\n');

  return [
    T({ en: '# Game build brief: ', id: '# Brief pembuatan game: ' }) + (c.title || T({ en: `${genreName} game`, id: `game ${genreName}` })),
    T({ en: 'Prompt framework: STAR (Situation, Task, Action, Result)', id: 'Kerangka prompt: STAR (Situation/Situasi, Task/Tugas, Action/Aksi, Result/Hasil)' }),
    '',
    T({ en: '## S: SITUATION', id: '## S: SITUASI' }),
    situation,
    '',
    T({ en: '## T: TASK', id: '## T: TUGAS' }),
    task,
    '',
    T({ en: '## A: ACTION', id: '## A: AKSI' }),
    action,
    '',
    T({ en: '## R: RESULT', id: '## R: HASIL' }),
    result,
  ].join('\n');
}

// ---------- Render ----------

let currentPrompt = '';

function highlight(text) {
  return escapeHtml(text)
    .replace(/^## S: .*$/m, (m) => `<span class="h-s">${m}</span>`)
    .replace(/^## T: .*$/m, (m) => `<span class="h-t">${m}</span>`)
    .replace(/^## A: .*$/m, (m) => `<span class="h-a">${m}</span>`)
    .replace(/^## R: .*$/m, (m) => `<span class="h-r">${m}</span>`)
    .replace(/^# .*$/m, (m) => `<span class="h-title">${m}</span>`);
}

function render() {
  const c = readConfig();
  $('#genreDesc').textContent = GENRES[c.genre].desc[uiLang];
  $('#objectiveWrap').hidden = !(c.purpose === 'education' || c.purpose === 'training');
  $('#customColors').hidden = c.palette !== 'custom';
  $('#swatches').innerHTML = c.colors.map((col) => `<span style="background:${escapeHtml(col)}"></span>`).join('');

  currentPrompt = buildPrompt(c);
  $('#prompt').innerHTML = highlight(currentPrompt);
  const words = currentPrompt.split(/\s+/).filter(Boolean).length;
  const s = UI[uiLang];
  $('#meta').textContent = `${words.toLocaleString(uiLang)} ${s.words} · ${currentPrompt.length.toLocaleString(uiLang)} ${s.chars}`;
  saveDraft();
}

// ---------- Presets & draft ----------

function setField(name, value) {
  const el = form.elements[name];
  if (!el) return;
  if (el instanceof RadioNodeList) {
    for (const r of el) r.checked = r.value === String(value);
  } else if (el.type === 'checkbox') {
    el.checked = !!value;
  } else {
    el.value = value;
  }
}

function applyPreset(p) {
  const promptLang = readConfig().promptLang;
  form.reset();
  setField('promptLang', promptLang);
  for (const [k, v] of Object.entries(p)) {
    if (k === 'features') continue;
    setField(k, v && typeof v === 'object' ? v[uiLang] : v);
  }
  for (const [id] of FEATURES) setField('f_' + id, p.features.includes(id));
  render();
}

const DRAFT_KEY = 'gamePromptStudio.draft.v2';
const UI_LANG_KEY = 'gamePromptStudio.uiLang';

function saveDraft() {
  try {
    const data = {};
    for (const el of form.elements) {
      if (!el.name) continue;
      if (el.type === 'checkbox') data[el.name] = el.checked;
      else if (el.type === 'radio') { if (el.checked) data[el.name] = el.value; }
      else data[el.name] = el.value;
    }
    localStorage.setItem(DRAFT_KEY, JSON.stringify(data));
  } catch (_) { /* storage unavailable: ignore */ }
}

function loadDraft() {
  try {
    const data = JSON.parse(localStorage.getItem(DRAFT_KEY) || 'null');
    if (!data) return false;
    for (const [k, v] of Object.entries(data)) setField(k, v);
    return true;
  } catch (_) { return false; /* ignore corrupt or blocked storage */ }
}

function savedUiLang() {
  try {
    const v = localStorage.getItem(UI_LANG_KEY);
    if (v === 'en' || v === 'id') return v;
  } catch (_) { /* ignore */ }
  return (navigator.language || '').toLowerCase().startsWith('id') ? 'id' : 'en';
}

// ---------- Actions ----------

let toastTimer;
function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 1800);
}

async function copyText(text, msg) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    // Fallback for non-secure contexts (e.g. opened via file://)
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
  }
  toast(msg);
}

function download() {
  const c = readConfig();
  const name = (c.title || 'game-prompt').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'game-prompt';
  const blob = new Blob([currentPrompt], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${name}-star-prompt-${c.promptLang}.txt`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/** Switch the interface language; the prompt language follows unless the user picks it separately afterwards. */
function switchUiLang(lang) {
  applyUiLang(lang);
  setField('promptLang', lang);
  try { localStorage.setItem(UI_LANG_KEY, lang); } catch (_) { /* ignore */ }
  render();
}

// ---------- Uploaded context (client-side parsing) ----------

let uploadedContextText = '';

async function parsePDF(buf) {
  const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
  const pages = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    pages.push(content.items.map(it => it.str).join(' '));
  }
  return pages.join('\n\n');
}

async function parseDOCX(buf) {
  const result = await mammoth.extractRawText({ arrayBuffer: buf });
  return result.value;
}

async function parseXLSX(buf) {
  const wb = XLSX.read(buf, { type: 'array' });
  const parts = [];
  for (const name of wb.SheetNames) {
    parts.push('[Sheet: ' + name + ']');
    const csv = XLSX.utils.sheet_to_csv(wb.Sheets[name]);
    parts.push(csv);
  }
  return parts.join('\n');
}

async function parsePPTX(buf) {
  const zip = await JSZip.loadAsync(buf);
  const slides = [];
  let i = 1;
  while (true) {
    const entry = zip.file('ppt/slides/slide' + i + '.xml');
    if (!entry) break;
    const xml = await entry.async('text');
    const texts = [];
    xml.replace(/<a:t>([\s\S]*?)<\/a:t>/g, (_, t) => { texts.push(t); });
    if (texts.length) slides.push('[Slide ' + i + ']\n' + texts.join('\n'));
    i++;
  }
  return slides.join('\n\n');
}

async function parseFile(file) {
  const buf = await file.arrayBuffer();
  const ext = file.name.split('.').pop().toLowerCase();
  if (ext === 'pdf') return parsePDF(buf);
  if (ext === 'docx' || ext === 'doc') return parseDOCX(buf);
  if (ext === 'xlsx' || ext === 'xls') return parseXLSX(buf);
  if (ext === 'pptx' || ext === 'ppt') return parsePPTX(buf);
  return '[Unsupported: ' + file.name + ']';
}

(function initUpload() {
  const dropZone = $('#dropZone');
  const fileInput = $('#fileInput');
  const fileList = $('#fileList');
  const preview = $('#contextPreview');
  const textEl = $('#contextText');
  const clearBtn = $('#clearContextBtn');
  const useWrap = $('#useContextWrap');

  function showContext(text) {
    uploadedContextText = text;
    textEl.textContent = text.length > 3000 ? text.slice(0, 3000) + '\n…' : text;
    preview.hidden = false;
    useWrap.hidden = false;
    render();
  }

  function clearContext() {
    uploadedContextText = '';
    fileInput.value = '';
    fileList.hidden = true;
    fileList.innerHTML = '';
    preview.hidden = true;
    useWrap.hidden = true;
    render();
  }

  async function handleFiles(files) {
    if (!files.length) return;
    fileList.hidden = false;
    fileList.innerHTML = Array.from(files).map(f =>
      '<span class="file-chip">' + escapeHtml(f.name) + '</span>'
    ).join('');
    dropZone.querySelector('.drop-text').textContent = UI[uiLang].contextParsing || '...';
    try {
      const parts = [];
      for (const f of files) {
        const text = await parseFile(f);
        if (text) parts.push('--- ' + f.name + ' ---\n' + text);
      }
      showContext(parts.join('\n\n'));
    } catch (e) {
      textEl.textContent = '[Error: ' + e.message + ']';
      preview.hidden = false;
    }
    applyUiLang(uiLang);
  }

  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => handleFiles(fileInput.files));
  clearBtn.addEventListener('click', clearContext);

  dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
  });
})();

// ---------- Init ----------

applyUiLang(savedUiLang());
if (!loadDraft()) setField('promptLang', uiLang);
render();

form.addEventListener('input', render);
form.addEventListener('change', render);
form.addEventListener('submit', (e) => e.preventDefault());
$('#copyBtn').addEventListener('click', () => copyText(currentPrompt, UI[uiLang].toastCopied));
$('#downloadBtn').addEventListener('click', download);
document.querySelectorAll('[data-ui-lang]').forEach((b) =>
  b.addEventListener('click', () => switchUiLang(b.dataset.uiLang))
);
document.querySelectorAll('[data-preset]').forEach((b) =>
  b.addEventListener('click', () => {
    const p = PRESETS[b.dataset.preset];
    applyPreset(p);
    toast(UI[uiLang].toastPreset(p.title[uiLang]));
  })
);
$('#followups').addEventListener('click', (e) => {
  if (e.target.tagName === 'LI') copyText(e.target.textContent, UI[uiLang].toastFollow);
});

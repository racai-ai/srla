<?php

exec("clear");
// Packs size
$sz1 = 5;
$sz2 = 3;
$sz3 = 10;

$dataDir = "data";

$wordListFile = "word.list";
$phone0ListFile = "phone0.list";
$phone1ListFile = "phone1.list";
$dictListFile = "blizzard_toshiba.list";
$dictLanguageFile = "blizzard_toshiba.dict";
$word0MLFFile = "word0.mlf";
$phone0MLFFile = "phone0.mlf";
$mkPhones0LEDFile = "resources/mkPhones0.led"; //*
$trainListFile = "train.scp";
$trainCFGFile = "resources/train.cfg"; //*
$trainMFCListFile = "train.mfc.scp";
$compCFGFile = "resources/comp.cfg";


echo PHP_EOL . "==================================================" . PHP_EOL;
echo "START voice aligner ...";
echo PHP_EOL . "--------------------------------------------------" . PHP_EOL;


if (!isValidCorpus($dataDir)) {
    echo "Corpusul din directorul $dataDir nu este valid ... exit" . PHP_EOL;
    exit();
}

initDirectories();
createWordListFile($dataDir, $wordListFile);
createDictionary($wordListFile, $dictLanguageFile);
createPhoneDictListFiles($wordListFile, $phone0ListFile, $dictListFile, $dictLanguageFile);
createPhoneDictListFiles($wordListFile, $phone1ListFile, $dictListFile, $dictLanguageFile);
fixPhoneListFiles($phone1ListFile, $phone0ListFile);
createWord0MLFFile($dataDir, $word0MLFFile);
createPhoneMKPhoneDictMLFFiles($dataDir, $dictListFile, $phone0MLFFile, $mkPhones0LEDFile, $word0MLFFile);
createTrainingList($dataDir, $trainListFile);
createMFCCs($trainCFGFile, $trainListFile);

echo "Initializing the model...";
echo PHP_EOL . "--------------------------------------------------" . PHP_EOL;

sleep(10); // se pare ca nu toate fisierele ajung in lista...poate nu sunt inca scrise
exec("sync");

createTrainMFCListFile($dataDir, $trainMFCListFile);
initHMM($compCFGFile, $trainMFCListFile);

echo "Beginning the first round of estimation..." . PHP_EOL;
for ($i = 1; $i <= $sz1; $i++) {
    echo "Iteration $i ..." . PHP_EOL;
    $prev = $i - 1;
    exec("HERest -C $compCFGFile -I $phone0MLFFile -t 250.0 150.0 1000.0 -S $trainMFCListFile -H hmm$prev/macros -H hmm$prev/hmmdefs -M hmm$i/ phone0.list");
}

echo "Fixing the silence models..." . PHP_EOL;

$sz = $sz1;
// creare hmmdefs
$lastFile = "hmm$sz/hmmdefs";
$lines = explode(PHP_EOL, file_get_contents($lastFile));
array_pop($lines); // sterge ultimul rand ca este gol
$silLines = array();
$copy = false;
$count = 0;
foreach ($lines as $line) {
    if (!$copy && strpos($line, "\"sil\"") > 0)
        $copy = true;
    if ($copy) {
        if ($count <= 20) {
            if ($count == 0) {
                $silLines[] = "~h \"sp\"";
                $silLines[] = "<BEGINHMM>";
                $silLines[] = "<NUMSTATES> 3";
            } else if ($count >= 3 && $count <= 8)
                $silLines[] = $line;
        } else {
            if ($count == 21) {
                $silLines[] = "<TRANSP> 3";
            } else if ($count < 25) {
                $tokens = explode(" ", $line);
                $items = 0;
                $matLine = "";
                for ($i = 0; $i < count($tokens) && $items < 3; $i++)
                    if (strlen($tokens[$i]) > 0) {
                        $matLine.=" " . $tokens[$i];
                        $items++;
                    }
                $silLines[] = $matLine;
            } else if ($count == 25) {
                $silLines[] = "<ENDHMM>";
            }
        }
        $count++;
    }
}
foreach ($silLines as $line)
    $lines[] = $line;
$lines[] = "";
$szpp = $sz + 1;
$lastFile = "hmm$szpp/hmmdefs";
file_put_contents($lastFile, implode(PHP_EOL, $lines));

// copy macros
$src = "hmm$sz/macros";
$dst = "hmm$szpp/macros";
copy($src, $dst);

$sz++;
$szpp = $sz + 1;
//HHEd -H hmm4/macros -H hmm4/hmmdefs -M hmm5 sil.hed phone1.list
exec("HHEd -H hmm$sz/macros -H hmm$sz/hmmdefs -M hmm$szpp resources/sil.hed $phone1ListFile");

//HLEd -l 'data/' -d dict.list -i phone1.mlf mkPhones1.led word0.mlf
echo "HLEd -l '$dataDir/' -d $dictListFile -i phone1.mlf resources/mkPhones1.led $word0MLFFile\n";
exec("HLEd -l '$dataDir/' -d $dictListFile -i phone1.mlf resources/mkPhones1.led $word0MLFFile");
$lines = explode(PHP_EOL, file_get_contents("phone1.mlf"));
$newLines = array();
foreach ($lines as $line)
    if (startsWith($line, "\"data")) {
        $newLines[] = "\"mfc" . substr($line, 5);
    } else {
        $newLines[] = $line;
    }

file_put_contents("phone1.mlf", implode(PHP_EOL, $newLines));


echo "Beginning the second round of estimation..." . PHP_EOL;
for ($i = 0; $i < $sz2; $i++) {
    $cur = $i + $sz1 + 2;
    echo "Iteration $cur ..." . PHP_EOL;
    $prev = $cur - 1;
    exec("HERest -C $compCFGFile -I phone1.mlf -t 250.0 150.0 1000.0 -S $trainMFCListFile -H hmm$prev/macros -H hmm$prev/hmmdefs -M hmm$cur/ phone1.list");
}

file_put_contents($dictListFile, file_get_contents($dictListFile) . "sil\tsil\n");

// Dublare cuvinte dict.list cu si fara sp la sfarsit
$lines = explode(PHP_EOL, file_get_contents($dictListFile));
$newLines = array();
foreach ($lines as $line) {
    if (!startsWith($line, "sil") && strlen($line) > 0) {
        $newLines[] = substr($line, 0, strlen($line) - 3);
    }
    $newLines[] = $line;
}
file_put_contents("dict.list", implode(PHP_EOL, $newLines));

echo "Realigning data..." . PHP_EOL;
//HVite -o SWT -b sil -C comp.cfg -a -H hmm7/macros -H hmm7/hmmdefs -i phone2.mlf -m -t 250.0 -y lab -I word0.mlf -S train.scp -L data/ dict.list phone1.list
echo $cur . PHP_EOL;
exec("HVite -o SWT -b sil -C resources/comp.cfg -a -H hmm$cur/macros -H hmm$cur/hmmdefs -i phone2.mlf -m -t 250.0 -y lab -I $word0MLFFile -S $trainMFCListFile -L $dataDir/ dict.list $phone1ListFile");

// Dublare cuvinte dict.list cu si fara sp la sfarsit
$lines = explode(PHP_EOL, file_get_contents("phone2.mlf"));
$newLines = array();
foreach ($lines as $line) {
    if (startsWith($line, "\"mfc")) {
        $newLines[] = substr($line, 1, strlen($line) - 5) . "mfc";
    }
}
$newLines[] = "";
file_put_contents($trainMFCListFile, implode(PHP_EOL, $newLines));


echo "Beginning the third round of estimation..." . PHP_EOL;
for ($i = 1; $i <= $sz3; $i++) {
    $cur = $cur + 1;
    echo "Iteration $cur ..." . PHP_EOL;
    $prev = $cur - 1;
    exec("HERest -C $compCFGFile -I phone2.mlf -t 250.0 150.0 1000.0 -S $trainMFCListFile -H hmm$prev/macros -H hmm$prev/hmmdefs -M hmm$cur/ phone1.list");
}

echo "Printing out final alignments...".PHP_EOL;
exec("HVite -o SM -b sil -C resources/comp.cfg -a -H hmm$cur/macros -H hmm$cur/hmmdefs -i word1.mlf -m -t 250.0 -y lab -I word0.mlf -S $trainMFCListFile -L $dataDir/ dict.list phone1.list");

echo "Extracting .phs files...".PHP_EOL;
extract_phs("word1.mlf", "aligned");

/*
  echo Printing out final alignments..
  HVite -o SM -b sil -C comp.cfg -a -H hmm9/macros -H hmm9/hmmdefs -i word1.mlf -m -t 250.0 -y lab -I word0.mlf -S train.scp -L data/ dict.list phone1.list
  ./wordLine word1.mlf
  ./textGrid word1.mlf
 */

//clear();


echo PHP_EOL . "--------------------------------------------------" . PHP_EOL;
echo "Voice aligner END";
echo PHP_EOL . "==================================================" . PHP_EOL . PHP_EOL;

function isValidCorpus() {
    global $dataDir;
    foreach (glob("$dataDir/*.lab") as $file) {
        $wav = substr($file, 0, strlen($file) - 3) . "wav";
        if (!file_exists($wav)) {
            unlink($file);
        }
    }
    foreach (glob("$dataDir/*.wav") as $file) {
        $lab = substr($file, 0, strlen($file) - 3) . "lab";
        if (!file_exists($lab)) {
            unlink($file);
        }
    }
    return true;
}

function extract_phs($filename, $folder) {
    $handle = fopen($filename, "r");
    $output = FALSE;
    if ($handle) {
        while (($line = fgets($handle)) !== false) {
            if (startsWith($line, "#")) {
                continue;
            } else if (startsWith($line, "."))
                continue;
            if (startsWith($line, "\"")) {
                if ($output != FALSE) {
                    fclose($output);
                }
                $output = fopen("aligned" . substr($line, 4, strlen($line) - 10) . ".phs", "w");
            } else {
                fputs($output, str_replace(" sil sil", " pau pau", $line));
            }
        }
        if ($output != FALSE) {
            fclose($output);
        }
        fclose($handle);
    } else {
        // error opening the file.
    }
}

function initDirectories() {
    global $sz1, $sz2, $sz3;
    echo "\tCreating directories ... ";
    if (!tryCreateDir("mfc")) {
        return false;
    }
    for ($i = 0; $i < $sz1 + $sz2 + $sz3 + 2; $i++) {
        if (!tryCreateDir("hmm$i")) {
            return false;
        }
    }
    if (!tryCreateDir("aligned")) {
        return false;
    }
    if (!tryCreateDir("textGrids")) {
        return false;
    }
    echo "ok." . PHP_EOL;
    return true;
}

function createWordListFile($dir, $wordListFile) {
    echo "\tCreating $wordListFile file ... ";
    $words = array();
    foreach (glob("$dir/*.lab") as $file) {
        foreach (preg_split("/[\s]+/", file_get_contents($file), -1, PREG_SPLIT_NO_EMPTY) as $word) {
            $words[] = $word;
        }
    }
    $words = array_unique($words);
    sort($words);
    file_put_contents($wordListFile, implode(PHP_EOL, $words));
    echo "ok." . PHP_EOL;
}

function createDictionary($wordListFile, $dictLanguageFile) {
    echo "\tCreating $dictLanguageFile file from $wordListFile file ... ";
    chdir("tools/lts");
    exec("java -jar MLPLA_LTS.jar --file ../../$wordListFile ../../$dictLanguageFile");
    chdir("../..");
    $line=exec("python sort.py $dictLanguageFile > $dictLanguageFile.sorted");
    $line=exec ("cp $dictLanguageFile.sorted $dictLanguageFile");
    echo "ok" . PHP_EOL;
}

function createPhoneDictListFiles($wordListFile, $phoneListFile, $dictListFile, $dictLanguageFile) {
    echo "\tCreating $phoneListFile and $dictListFile files ... ";
    echo "HDMan -m -w $wordListFile -n $phoneListFile $dictListFile $dictLanguageFile\n";
    exec("HDMan -m -w $wordListFile -n $phoneListFile $dictListFile $dictLanguageFile");
    echo "ok" . PHP_EOL;
}

function fixPhoneListFiles($phone1ListFile, $phone0ListFile) {
    echo "\tCreating $phone0ListFile file and fix $phone1ListFile file ... ";

    $ph0s = array();
    $ph1s = array_filter(explode(PHP_EOL, file_get_contents($phone1ListFile)), "noEmpty");

    foreach ($ph1s as $ph1) {
        if ($ph1!="sp") {
            echo "'".$ph1."'";
            $ph0s[] = $ph1;
        }
    }

    array_push($ph0s, "sil");
    //$ph0s[] = "sil";
    $ph0s[] = "";
    $ph0s = array_unique($ph0s);

    $ph1s[] = "sil";
    $ph1s[] = "";
    $ph1s = array_unique($ph1s);

    file_put_contents($phone0ListFile, implode(PHP_EOL, $ph0s));
    file_put_contents($phone1ListFile, implode(PHP_EOL, $ph1s));
    echo "ok" . PHP_EOL;
}

function createWord0MLFFile($dir, $word0MLFFile) {
    echo "\tCreating master label file $word0MLFFile ... ";
    $mlf = array("#!MLF!#");
    foreach (glob("$dir/*.lab") as $file) {
        $mlf[] = "\"$file\"";
        foreach (preg_split("/[\s]+/", file_get_contents($file), -1, PREG_SPLIT_NO_EMPTY) as $word) {
	    if (is_numeric($word)){
		$word="_".$word;
	    }
            $mlf[] = $word;
        }
        $mlf[] = ".";
    }
    $mlf[] = "";
    file_put_contents($word0MLFFile, implode(PHP_EOL, $mlf));
    echo "ok" . PHP_EOL;
}

function createPhoneMKPhoneDictMLFFiles($dir, $dictListFile, $phone0MLFFile, $mkPhones0LEDFile, $word0MLFFile) {
    echo "\tCreating $phone0MLFFile MKPHone MLF files ... ";
    echo "HLEd -l 'mfc/' -d $dictListFile -i $phone0MLFFile $mkPhones0LEDFile $word0MLFFile\n";
    exec("HLEd -l 'mfc/' -d $dictListFile -i $phone0MLFFile $mkPhones0LEDFile $word0MLFFile");
    echo "ok" . PHP_EOL;
}

function createTrainingList($dir, $trainListFile) {
    echo "\tCreating training list file $trainListFile ... ";
    $strs = array();
    foreach (glob("$dir/*.wav") as $file) {
        $len = strlen($file);
        $strs[] = $file . " mfc/" . substr($file, 5, $len - 5 - 3) . "mfc";
    }
    $strs[] = "";
    file_put_contents($trainListFile, implode(PHP_EOL, $strs));
    echo "ok" . PHP_EOL;
}

function createMFCCs($trainCFGFile, $trainListFile) {
    echo "\tCreating MFCCs using $trainCFGFile, $trainListFile files ... ";
    exec("HCopy -T 1 -C $trainCFGFile -S $trainListFile");
    echo "ok" . PHP_EOL;
}

function createTrainMFCListFile($dataDir, $trainMFCListFile) {
    echo "\tCreating training mfc file $trainMFCListFile ... ";
    $strs = array();
    foreach (glob("mfc/*.mfc") as $file) {
        $strs[] = $file;
    }
    $strs[] = "";
    file_put_contents($trainMFCListFile, implode(PHP_EOL, $strs));
    echo "ok" . PHP_EOL;
}

function initHMM($compCFGFile, $trainMFCListFile) {
    echo "\tInitialize HMM ... ";
    exec("HCompV -C $compCFGFile -f 0.01 -S $trainMFCListFile -M hmm0/ resources/proto");
    $str = array();
    $str[] = file_get_contents("resources/mactmp.txt");
    $str[] = file_get_contents("hmm0/vFloors");
    file_put_contents("hmm0/macros", implode(PHP_EOL, $str));

    $protos = explode(PHP_EOL, file_get_contents("hmm0/proto"));
    $phs = explode(PHP_EOL, file_get_contents("phone0.list"));
    $out = array();
    foreach ($phs as $ph)
        if (strlen($ph) > 0)
            foreach ($protos as $proto)
                if (strlen($proto) > 0)
                    $out[] = str_replace("\"proto\"", "\"$ph\"", $proto);
    file_put_contents("hmm0/hmmdefs", implode(PHP_EOL, $out));
    echo "ok" . PHP_EOL;
}

function clear() {
    echo "\tRemove directories ... ";
    foreach (scandir(".") as $file) {
        if (is_dir($file) && startsWith(basename($file), "hmm")) {
            if (!rmdir($file)) {
                return false;
            }
        }
    }
    if (!rmdir("mfc")) {
        return false;
    }
    if (!rmdir("aligned")) {
        return false;
    }
    if (!rmdir("textGrids")) {
        return false;
    }
    echo "ok" . PHP_EOL;
    return true;
}

function startsWith($haystack, $needle) {
    // search backwards starting from haystack length characters from the end
    return $needle === "" || strrpos($haystack, $needle, -strlen($haystack)) !== FALSE;
}

function endsWith($haystack, $needle) {
    // search forward starting from end minus needle length characters
    return $needle === "" || (($temp = strlen($haystack) - strlen($needle)) >= 0 && strpos($haystack, $needle, $temp) !== FALSE);
}

function tryCreateDir($dir) {
    if (!is_dir($dir)) {
        if (!mkdir($dir)) {
            return false;
        }
    }
    return true;
}

function noEmpty($str) {
    if (!$str)
        return false;
    if (strlen($str) == 0)
        return false;
    return true;
}
?>


<?php

$dir="/data/CORPUSURI/RADOR";


function startsWith($haystack, $needle)
{
     $length = strlen($needle);
     return (substr($haystack, 0, $length) === $needle);
}

function endsWith($haystack, $needle)
{
    $length = strlen($needle);
    if ($length == 0) {
        return true;
    }

    return (substr($haystack, -$length) === $needle);
}


$c=0;
if (is_dir($dir)) {
    if ($dh = opendir($dir)) {
        while (($file = readdir($dh)) !== false) {
            $fname=$dir."/".$file;
            if(is_file($fname) && endsWith($fname,".lab")){
		$text=file_get_contents($fname);
		file_put_contents("data/$file",$text);

		$fwav=substr($file,0,strlen($file)-4).".wav";
		file_put_contents("data/$fwav",file_get_contents($dir."/".$fwav));
		$c++;

		if($c>2000)break;
            }
        }
        closedir($dh);
    }
}



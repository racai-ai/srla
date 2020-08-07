<?php

$dir="data";


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


if (is_dir($dir)) {
    if ($dh = opendir($dir)) {
        while (($file = readdir($dh)) !== false) {
            $fname=$dir."/".$file;
            if(is_file($fname) && endsWith($fname,".txt")){
		$text=file_get_contents($fname);
		$text=mb_strtolower($text);
		$text=preg_replace("/[^a-zăîâșț]/"," ",$text);
		$text=preg_replace("/[ ]+/"," ",$text);
		//echo "$text\n";
		file_put_contents(substr($fname,0,strlen($fname)-4).".lab",trim($text));
            }
        }
        closedir($dh);
    }
}



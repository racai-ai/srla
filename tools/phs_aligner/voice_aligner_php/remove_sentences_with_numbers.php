<?php
function startsWith($haystack, $needle)
{
     $length = strlen($needle);
     return (substr($haystack, 0, $length) === $needle);
}

function endsWith($haystack, $needle)
{
    $length = strlen($needle);

    return $length === 0 || 
    (substr($haystack, -$length) === $needle);
}

$files=scandir ($argv[1]);
for ($i=0;$i<sizeof($files);$i++){
        $file=$files[$i];
        if (endsWith($file, ".lab")){
		$text=file_get_contents($argv[1]."/".$file);
		$rez=preg_match_all('!\d+!', $text, $matches);
		if ($rez!=0){
			//echo $file."\n";
			$lab=$argv[1]."/".$file;
			$wav=str_replace(".lab",".wav",$lab);
			echo "$wav\t$lab\n";
			$line=exec("rm $wav");
			$line=exec("rm $lab");
		}
	}
}
?>

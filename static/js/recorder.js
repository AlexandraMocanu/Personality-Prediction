var buttonRecord = document.getElementById("record");
var buttonStop = document.getElementById("stop");
var buttonPredict = document.getElementById("predict");

buttonStop.disabled = true;
buttonPredict.disabled = true;

buttonRecord.onclick = function() {
    // var url = window.location.href + "record_status";
    buttonRecord.disabled = true;
    buttonStop.disabled = false;
    
    // disable download link
    var downloadLink = document.getElementById("download");
    downloadLink.text = "";
    downloadLink.href = "";

    // XMLHttpRequest
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState == 4 && xhr.status == 200) {
            // alert(xhr.responseText);
        }
    }
    xhr.open("POST", "/record/record_status");
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.send(JSON.stringify({ status: "true" }));
};

buttonStop.onclick = function() {
    buttonRecord.disabled = false;
    buttonStop.disabled = true;

    buttonRecord.style.visibility.hidden = true;
    buttonStop.style.visibility.hidden = true;

    buttonPredict.disabled = false;

    // XMLHttpRequest
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState == 4 && xhr.status == 200) {
            // alert(xhr.responseText);

            // enable download link
            var downloadLink = document.getElementById("download");
            downloadLink.text = "Video saved! Please click \"Predict\" when ready.";
            path = records_folder + "/" + "video_"+username + ".avi"
            console.log(path)
            downloadLink.href = path; 
        }
    }
    xhr.open("POST", "/record/record_status");
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.send(JSON.stringify({ status: "false" }));
};
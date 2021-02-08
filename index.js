let net;
const imgEl = document.getElementById('img');
const container = document.getElementById('prediction');

async function app() {
  console.log('Cargando modelo mobilenet...');

  // Cargar el modelo.
  net = await mobilenet.load();
  console.log('Modelo cargado exitosamente');

  // Hacer una predicción a través de una imagen.
  var result = await net.classify(imgEl);
  console.log(result);
  displayPrediction()
  
}
app();

count = 0;
async function cambiarImagen(){
  count = count + 1;
  imgEl.src = "https://picsum.photos/200/300?random=" + count;
}

imgEl.onload = async function(){
  displayPrediction();
}

async function displayPrediction(){
  try{
    result = await net.classify(imgEl);
    console.log(result);
    container.innerHTML = "Predicción: "+ result[0].className;
  }catch(error){

  }
}
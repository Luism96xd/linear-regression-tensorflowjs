var output;
var container = document.getElementById('container');
var button = document.getElementById('run_button');
var input_predict = document.getElementById('predict');
var predecir_btn = document.getElementById('predict_btn');
var entradas = [];
var salidas = [];
var dimensiones = [];
var model;
var prediccion;
const alerts = document.getElementById('alerts');
const sampleButton = document.getElementById('sample');

function readFile(evt) {
    let file = evt.target.files[0];
    let reader = new FileReader();
    reader.onload = (e) => {
      // Cuando el archivo se terminó de cargar
      let lines = parseCSV(e.target.result);
      output = reverseMatrix(lines);
      console.log(output);
      displayHTMLTable(output);
    };
    // Leemos el contenido del archivo seleccionado
    reader.readAsBinaryString(file);
  }
  
  function parseCSV(text) {
    // Obtenemos las lineas del texto
    let lines = text.replace(/\r/g, '').split('\n');
    return lines.map(line => {
      // Por cada linea obtenemos los valores
      let values = line.split(',');
      return values;
    });
  }
  
  function reverseMatrix(matrix){
    let output = [];
    // Por cada fila
    matrix.forEach((values, row) => {
      // Vemos los valores y su posicion
      values.forEach((value, col) => {
        // Si la posición aún no fue creada
        if (output[col] === undefined) output[col] = [];
        output[col][row] = value;
      });
    });
    flash('Data uploaded successfully', 'success')
    return output;
  }

  function displayHTMLTable(results) {
    var table = "<table class='table'>";
    var data = results;

    for (i = 0; i < data.length; i++) {
        table += "<tr>";
        var row = data[i];
        var cells = row.join(",").split(",");

        for (j = 0; j < cells.length; j++) {
            table += "<td> ";
            table += cells[j];
            table += "</th>";
        }
        table += "</tr>";
    }
    table += "</table>";
    tabla = document.createElement('div');
    tabla.innerHTML = table;
    container.appendChild(tabla);

    for (i = 1; i < row.length; i++){
      entradas.push(Number.parseInt(output[0][i]));
      salidas.push(Number.parseInt(output[1][i]));
    }
    getData(entradas);

    dimensiones = [entradas.length, 1];
    console.log(dimensiones);
    console.log(entradas, typeof(entradas));
    console.log(salidas, typeof(salidas));
    console.log("X:"+ entradas + "\t Length: "+ dimensiones);
    console.log("Y:"+ salidas + "\t Length: "+ dimensiones);
}

document.getElementById('files').addEventListener('change', readFile, false);

button.addEventListener('click', learnLinear);

sampleButton.addEventListener('click', function(){
  entradas = [-1, 0, 1, 2, 3, 4, 5];
  salidas = [-3, -1, 1, 3, 5, 7, 8];
  dimensiones = [entradas.length, 1]
  getData(entradas);
});

predecir_btn.addEventListener('click', function(){
  if((input_predict.value != "" && model != undefined)){
    predecir(Number.parseInt(input_predict.value));
  }
  else{
    flash('Model has not been created yet','error');
  }
});
function flash(message, category){
  if (category == "success" && category != null){
    color = '#9BFF96';
  }else{
    color = '#ff969b';
  }
  const div = document.createElement('div');
  div.style.backgroundColor = color;
  div.classList.add('alert');
  div.innerText = message;
  div.innerHTML += `  <span class="closebtn" onclick="this.parentElement.style.display='none'; opacity: 0;">&times;</span>`;
  alerts.appendChild(div);
  setTimeout(function(){ alerts.removeChild(div); }, 2000);
}


async function learnLinear(){
  model = tf.sequential();
  model.add(tf.layers.dense({units: 1, inputShape: [1]}));//, activation="relu"));
  model.add(tf.layers.dense({units: 1, inputShape: [1]}));
  
  const learningRate = 0.00001;
  const optimizer = tf.train.sgd(learningRate);
  model.compile({
      loss: 'meanSquaredError',
      optimizer: tf.train.adam(0.0557)
  });

  var xs = tf.tensor2d(entradas, dimensiones);
  var ys = tf.tensor2d(salidas, dimensiones); 

  console.log(xs, typeof(xs));
  console.log(ys, typeof(ys))
  await model.fit(xs, ys, {epochs: 100,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mae'], 
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
  console.log("Modelo entrenado");
  testModel(model, getData());

  predecir(20);
}

function testModel(model, data){
  const [xs, preds] = tf.tidy(() => {
    
    const xs = tf.linspace(0, 150, 100);      
    const preds = model.predict(xs.reshape([100, 1]));
    console.log(xs.print())
    console.log(preds.print()); 
    return [xs.dataSync(),preds.dataSync()];     
  });
  
  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });
  console.log(predictedPoints); 
  const originalPoints = data;
  
  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data'}, 
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']}, 
    {
      xLabel: 'Tiempo',
      yLabel: 'Carga',
      height: 300
    }
  );
}

function getData(){
  const values = entradas.map((d,i) => ({
    x: entradas[i],
    y: salidas[i]
  }));
  
  console.log(values);

  tfvis.render.scatterplot(
    {name: 'Tiempo vs. Carga de la Batería'},
    {values}, 
    {
      xLabel: 'Tiempo',
      yLabel: 'Carga de la batería',
      height: 300
    }
  );
  return values;
}
async function predecir(n){
  const preds = await tf.tidy( () => {
    if(input_predict.value != ""){
      n = Number.parseInt(input_predict.value);
      var output = model.predict(tf.tensor2d([n], [1,1]));
      prediccion = Array.from(output.dataSync())[0];
      document.getElementById('output_field').innerHTML = `<b>Predicción: </b><br><p>${prediccion}</p>`;
      console.log(prediccion);
    }else{
      var output = model.predict(tf.tensor2d([n], [1,1]));
      prediccion = Array.from(output.dataSync())[0];
      document.getElementById('output_field').innerHTML = `<b>Predicción: </b><br><p>${prediccion}</p>`;
      console.log(prediccion);
    }
  });
}
const model = new Model();


document.getElementById('inference').addEventListener('click', async () => {
  result_label = model.inference();
  document.getElementById('result_txt').innerText=result_label
});

async function init() {
  try {
    await model.setup();
  } catch (e) {
    console.log(e);
    document.getElementById('result_txt').innerText="Model_keras load failed!!"
  }
}

// Initialize the application.
init();

class Model {
    constructor() {
        console.log('Model_keras constructor');
        this.MODEL_URL = 'webmod_keras/model.json';
    }

    inference(data) {
        console.log('Model_keras inference');
        try {
            // var xs = tf.tensor2d([pixels])
            var xs = tf.fill([1, 784], 0)
            var result = this.model.predict(xs);
            return result.dataSync()
        } catch (e) {
            console.log(e)
        }
    }

    async setup() {
        console.log('Model_keras setup');
        try {
            this.model = await tf.loadLayersModel(this.MODEL_URL);
            console.log('Model_keras is loadded');
        } catch (e) {
            console.log(e)
        }
    }

}

const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const bmp = require('bmp-js');
const util = require('util');

const readFile = util.promisify(fs.readFile);

const TRAIN_IMAGES_PATH = '/home/mout/Downloads/38300_58521_bundle_archive/SOCOFing/Real/';
const TRAIN_LABELS_FILE = '/home/mout/Downloads/38300_58521_bundle_archive/SOCOFing/Real/';
const TEST_IMAGES_FILE = '/home/mout/Downloads/38300_58521_bundle_archive/SOCOFing/Altered/Altered-Easy/';
const TEST_LABELS_FILE = '/home/mout/Downloads/38300_58521_bundle_archive/SOCOFing/Altered/Altered-Easy/';
const IMAGE_HEADER_MAGIC_NUM = 2051;
const IMAGE_HEADER_BYTES = 16;
const IMAGE_HEIGHT = 103;
const IMAGE_WIDTH = 96;
const IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
const LABEL_HEADER_MAGIC_NUM = 2049;
const LABEL_HEADER_BYTES = 8;
const LABEL_RECORD_BYTE = 1;
const LABEL_FLAT_SIZE = 10;

async function getAllImages(filepath, files) {
    return new Promise(resolve => {
        let bmpPromises = [];
        let bmpBuffers = [];
        let bmpBuffersLength = 0;
        files.forEach(file => {
            bmpPromises.push(readFile(`${filepath}${file}`));
        }) 
        Promise.all(bmpPromises)
        .then(buffers => {
            buffers.forEach(buffer => {
                const bmpData = bmp.decode(buffer);
                bmpBuffersLength += bmpData.data.length;
                bmpBuffers.push(bmpData.data);
                });
            // resolve(Buffer.concat(bmpBuffers, bmpBuffersLength));
            resolve(bmpBuffers);
        });
    })
}

async function getAllFiles(filepath) {
    return new Promise(resolve => {
        resolve(fs.readdirSync(filepath))
    });
}

async function loadImages(filepath) {
    const files = await getAllFiles(filepath)
    const buffer = await getAllImages(filepath, files);
    const recordBytes = IMAGE_HEIGHT * IMAGE_WIDTH;
    
    // let index = 0;
    // const images = [];
    // console.log()
    // while (index < buffer.byteLength - 1) {
    //     const array = new Float32Array(recordBytes);
    //     for(let i = 0; i < recordBytes; i++) {
    //         if (index >= buffer.byteLength) break;
    //         array[i] = buffer.readUInt8(index++) / 255;
    //     }
    //     images.push(array);
    // }
    return buffer;
}

async function loadLabels(filepath) {
    const files = await getAllFiles(filepath);
    const recordBytes = LABEL_RECORD_BYTE;

    const labels = [];

    files.forEach(file => {
        labels.push(parseInt(file));
    })
    return labels;
}

class DataSet {
    constructor() {
        this.dataset = null;
        this.trainSize = 0;
        this.testSize = 0;
        this.trainBatchIndex = 0;
        this.testBatchIndex = 0;
    }

    async loadData() {
        this.dataset = await Promise.all([
            loadImages(TRAIN_IMAGES_PATH), loadLabels(TRAIN_LABELS_FILE),
            loadImages(TEST_IMAGES_FILE), loadLabels(TEST_LABELS_FILE)
        ]);
        this.trainSize = this.dataset[0].length;
        this.testSize = this.dataset[2].length;
    }

    getTrainData() {
        return this.getData_(true);
    }

    getTestData() {
        return this.getData_(false);
    }

    getData_(isTrainingData) {
        let imagesIndex;
        let labelsIndex;
        if (isTrainingData) {
            imagesIndex = 0;
            labelsIndex = 1;
        } else {
            imagesIndex = 2;
            labelsIndex = 3;
        }
        const size = this.dataset[imagesIndex].length;
        tf.util.assert(
            this.dataset[labelsIndex].length === size,
            `Mismatch in the number of images (${size}) and ` +
            `the number of labels (${this.dataset[labelsIndex].length})`);

        const imagesShape = [size, IMAGE_HEIGHT, IMAGE_WIDTH, 1];
        const images = new Float32Array(tf.util.sizeFromShape(imagesShape));
        const labels = new Int32Array(tf.util.sizeFromShape([size, 1]));

        let imageOffset = 0;
        let labelOffset = 0;
        for (let i = 0; i < size; ++i) {
            images.set(this.dataset[imagesIndex][i], imageOffset);
            labels.set([this.dataset[labelsIndex][i]], labelOffset);
            imageOffset += 1;
            labelOffset += 1;
        }

        return {
            images: tf.tensor4d(images, imagesShape),
            labels: tf.oneHot(tf.tensor1d(labels, 'int32'), LABEL_FLAT_SIZE).toFloat()
        };
    }
}

module.exports = new DataSet();
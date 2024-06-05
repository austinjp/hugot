package pipelines

import (
	jsoniter "github.com/json-iterator/go"
	util "github.com/knights-analytics/hugot/utils"
	ort "github.com/yalue/onnxruntime_go"
)

// types

type TextSummarizationPipeline struct {
	BasePipeline
}

type TextSummarizationPipelineConfig struct {
}

type TextSummarizationPipelineBatchOutput struct{}

type SummarizationOutput struct {
	Words []string
}

type TextSummarizationOutput struct {
	SummarizationOutputs [][]SummarizationOutput
}

func (e *TextSummarizationPipelineBatchOutput) GetOutput() []any {
	out := make([]any, len(e.SummarizationOutputs))
	for i, summarizationOutput := range e.SummarizationOutputs {
		out[i] = any(summarizationOutput)
	}
	return out
}

func NewTextSummarizationPipeline(config PipelineConfig[*TextSummarizationPipeline], ortOptions *ort.SessionOptions) (*TextSummarizationPipeline, error) {
	pipeline := &TextSummarizationPipeline{}
	pipeline.ModelPath = config.ModelPath
	pipeline.PipelineName = config.Name
	pipeline.OrtOptions = ortOptions
	pipeline.OnnxFilename = config.OnnxFilename
	for _, o := range config.Options {
		o(pipeline)
	}

	// load json model config and set pipeline settings
	configPath := util.PathJoinSafe(config.ModelPath, "config.json")
	pipelineInputConfig := TokenClassificationPipelineConfig{}
	mapBytes, err := util.ReadFileBytes(configPath)
	if err != nil {
		return nil, err
	}

	err = jsoniter.Unmarshal(mapBytes, &pipelineInputConfig)
	if err != nil {
		return nil, err
	}

	// load onnx model
	errModel := pipeline.loadModel()
	if errModel != nil {
		return nil, errModel
	}

	// the dimension of the output is taken from the output meta.
	// pipeline.OutputDim = int(pipeline.OutputsMeta[0].Dimensions[2])

	// err = pipeline.Validate()
	// if err != nil {
	// 	return nil, err
	// }

	return pipeline, nil
}

func (p *TextSummarizationPipeline) Validate() error {
	return nil
}

func (p *TextSummarizationPipeline) Run(inputs []string) (PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

func (p *TextSummarizationPipeline) RunPipeline(inputs []string) (*TextSummarizationPipelineBatchOutput, error) {
	// batch := p.Preprocess(inputs)
	// batch, errForward := p.Forward(batch)
	// if errForward != nil {
	// 	return "", errForward
	// }
	// return "", nil
	return &TextSummarizationPipelineBatchOutput{}, nil
}

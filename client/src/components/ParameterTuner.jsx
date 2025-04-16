// client/src/components/ParameterTuner.jsx
import Slider from '@mui/material/Slider'

const ParameterTuner = ({ onUpdate }) => {
    const [params, setParams] = useState({
        learningRate: 0.001,
        batchSize: 32,
        attentionHeads: 3
    })

    const handleCommit = () => {
        fetch('/api/train', {
            method: 'POST',
            body: JSON.stringify(params)
        })
    }

    return (
        <div className="parameter-panel">
            <Slider
                value={params.learningRate}
                min={0.0001}
                max={0.01}
                step={0.0001}
                onChange={(e, v) => setParams({ ...params, learningRate: v })}
            />
            <Button onClick={handleCommit}>
                Apply New Parameters
            </Button>
        </div>
    )
}

// client/src/components/TelomereVisualizer.jsx
import { useThree } from '@react-three/fiber'
import { Suspense } from 'react'
import { ScatterChart } from '@visx/xychart'

const TelomereVisualizer = ({ data }) => {
    return (
        <Suspense fallback={<Loader />}>
            <div className="visualization-container">
                <ScatterChart
                    width={800}
                    height={500}
                    data={data}
                    xScale={{ type: 'time' }}
                    yScale={{ type: 'linear' }}
                >
                    <Axis orientation="bottom" />
                    <Axis orientation="left" />
                    <LineSeries
                        dataKey="telomere"
                        xAccessor={d => new Date(d.timestamp)}
                        yAccessor={d => d.length}
                    />
                    <HeatmapOverlay
                        data={attentionData}
                        bandwidth={0.05}
                    />
                </ScatterChart>
            </div>
        </Suspense>
    )
}

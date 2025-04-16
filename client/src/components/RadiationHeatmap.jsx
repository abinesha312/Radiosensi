// client/src/components/RadiationHeatmap.jsx
import { useThree } from '@react-three/fiber'
import { Heatmap } from '@visx/heatmap'

const RadiationHeatmap = ({ telomereData }) => {
    const { width, height } = useThree().size

    return (
        <Heatmap
            data={telomereData}
            x={d => d.time}
            y={d => d.length}
            color={d => d.attention}
            width={width}
            height={height}
            binSize={15}
        >
            {heatmap => (
                <image
                    width={width}
                    height={height}
                    href={heatmap.render().toDataURL()}
                />
            )}
        </Heatmap>
    )
}

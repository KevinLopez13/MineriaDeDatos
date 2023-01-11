// yarn add @nivo/core @nivo/heatmap
import React from 'react';
import { ResponsiveHeatMap } from '@nivo/heatmap'


export default function HeatMap ({ data = [] }){
    
    if (data.length === 0){
        return null;
    }

    return (
        <div className='mt-3' style={{height: '350px', width:'auto', textAlign:"center"}}>
        <h6>Mapa de calor</h6>
        <ResponsiveHeatMap
            data={data.dataGraph}
            margin={{ top: 10, right: 90, bottom: 100, left: 90 }}
            valueFormat=">-.2r"
            axisTop={null}
            axisBottom={{
                tickSize: 5,
                tickPadding: 5,
                tickRotation: -40,
                legend: '',
                legendPosition: 'middle',
                legendOffset: 36
            }}
            axisLeft={{
                tickSize: 5,
                tickPadding: 5,
                tickRotation: 0,
                legend: '',
                legendPosition: 'middle',
                legendOffset: -72
            }}
            colors={{
                type: 'diverging',
                scheme: 'red_yellow_blue',
                divergeAt: 0.5,
                minValue: 1,
                maxValue: -1
            }}
            emptyColor="#fff"
            legends={[
                {
                    anchor: 'right',
                    translateX: 30,
                    translateY: -10,
                    length: 350,
                    thickness: 8,
                    direction: 'column',
                    tickPosition: 'after',
                    tickSize: 3,
                    tickSpacing: 4,
                    tickOverlap: false,
                    tickFormat: '>-.2s',
                    title: 'Value →',
                    titleAlign: 'start',
                    titleOffset: 4
                }
            ]}
        />
        </div>
    )
}
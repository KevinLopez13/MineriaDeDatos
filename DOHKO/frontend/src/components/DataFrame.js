
import React, { useState, useEffect } from 'react';
import Table from 'react-bootstrap/Table';

export default function DataFrame({data=[]}) {
    
    if (data.length === 0){
        return null;
    }

    return (
        <div style={{maxHeight:"350px", overflow:"auto"}} >

            <Table responsive striped bordered hover size="sm" style={{fontSize:14}} >
                <thead>
                <tr>
                    {data.cols.map((column, index) => (
                    <th key={index} style={{textAlign:"center"}}>{column}</th>
                    ))}
                </tr>
                </thead>
                <tbody>            
                {data.rows.map((row, index) => (
                    <tr>
                    {row.map((col, index) => (
                    <td key={index} style={{textAlign:"center"}}>{col}</td>
                    ))}
                    </tr>
                ))}
                
                </tbody>
            </Table>
        </div>
    );
}
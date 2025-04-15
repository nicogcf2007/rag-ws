import React from 'react';

const TestComponent: React.FC = () => {
  return (
    <div className="p-4 bg-base-200 min-h-screen">
      <div className="max-w-md mx-auto bg-base-100 shadow-xl rounded-lg p-6">
        <h1 className="text-2xl font-bold text-primary mb-4">Test de Tailwind y DaisyUI</h1>
        <p className="text-base-content mb-4">Si puedes ver este texto con estilos, Tailwind y DaisyUI están funcionando correctamente.</p>
        <button className="btn btn-primary">Botón de Prueba</button>
      </div>
    </div>
  );
};

export default TestComponent;

using BookRent.Orchestrator.Api.Requests;
using BookRent.Orchestrator.Services.Interfaces;
using Microsoft.AspNetCore.Mvc;

namespace BookRent.Orchestrator.Api;

internal static class EmployeeOrchestrationSagas
{

    public static async Task<IResult> RemoveBookFromCatalogSaga([FromBody] RemoveFromCatalogRequest request, ICatalogService service)
    {
        return await service.RemoveFromCatalogSaga(request);
    }
    
    
}
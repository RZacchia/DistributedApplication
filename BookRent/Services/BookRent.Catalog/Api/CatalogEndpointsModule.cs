using BookRent.Catalog.Infrastructure.Interfaces;

namespace BookRent.Catalog.Api;

internal static class CatalogEndpointsModule
{
    internal static void MapCatalogEndpoints(this IEndpointRouteBuilder app)
    {
        RouteGroupBuilder group = app.MapGroup("/catalog");

        group.MapGet("/books", CatalogCustomerEndpoints.GetAllBooks);
        group.MapGet("/books/search", CatalogCustomerEndpoints.SearchBook);
        group.MapGet("/book/{id:guid}", CatalogCustomerEndpoints.GetBook);
        group.MapPost("/addBook", CatalogEmployeeEndpoints.AddBook);
        group.MapPost("/removeBook/{id:guid}", CatalogEmployeeEndpoints.RemoveBook);
        group.MapPost("/updateBook", CatalogEmployeeEndpoints.UpdateBook);
    }
}